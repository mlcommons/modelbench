""" A basic library for multithreaded pipelines written so that users don't have to understand Python concurrency.

 In this code, a Pipeline is a collection of PipelineSegments, each of which runs in its own threads. A Source
 is the head of the pipe; a Pipe is any number of middle processing stages; a Sink is the last stage. A Pipe
 can have multiple threads, which is best used when when python is waiting for something external to happen,
 like a remote API call.

 To create your own pipeline, you'll need to subclass Source, Pipe (possibly more than once), and Sink, implementing
 the abstract method on each. Then you can put them together like this:

    p = Pipeline(
        MySource(),
        MyPipe(),
        MySink()
    )
    p.run()

That will run every item produced by the source, passing the output of each stage along to the next.

What gets passed along? Whatever your stages produce. There's no type requirement. So for example:

    class MySource(Source):

        def new_item_iterable(self):
            return [1, 2, 3]


    class MyPipe(Pipe):

        def handle_item(self, item):
            return item * 2


    class MySink(Sink):

        def handle_item(self, item):
            print(item)

Will print 2, 4, and 6 when run.

That example just has one thread per PipeSegment. If you'd like more workers in a given Pipe, you might do something
like:

    class MyPipe(Pipe):

        def __init__(self, ):
            super().__init__(thread_count=8)

Note that this generally only helps when the code is waiting on the network or other I/O. Python's Global Intepreter
Lock (GIL) means that generally only one bit of python is running at once.

 """

import datetime
import queue
import sys
import threading
from abc import abstractmethod, ABC
from queue import Queue
from threading import Event, Thread
from typing import Any, Callable, Iterable, Optional

import diskcache  # type: ignore


class PipelineSegment(ABC):
    """A segment of a Pipeline used for parallel processing."""

    default_timeout = 0.1

    def __init__(self):
        super().__init__()
        self._work_done = Event()
        self._upstream: Optional[PipelineSegment] = None
        self._queue: Queue = Queue()
        self.completed = 0
        self._debug_enabled = False
        self._thread_id = 0

    def start(self):
        pass

    def downstream_put(self, item: Any):
        """Enqueue an item for the next segment in the pipeline to take."""
        self._queue.put(item)

    def upstream_get(self, timeout=None):
        """Dequeue an item from the previous segment in the pipeline"""
        if timeout is None:
            timeout = PipelineSegment.default_timeout
        if self._upstream is None:
            raise (ValueError(f"{self} doesn't have an upstream connection"))
        self._debug("getting from upstream")

        try:
            item = self._upstream._queue.get(timeout=timeout)
        except queue.Empty:
            self._debug("get was empty")
            raise

        self._debug("got item")

        return item

    def upstream_task_done(self):
        self._upstream._queue.task_done()

    def done(self):
        if self._upstream and not self._upstream.done():
            return False

        return self._work_done.is_set() and self._queue.empty()

    def join(self):
        self._debug(
            f"joining queue {self._queue}: {self._queue.qsize()} {self._queue.unfinished_tasks} {self._queue.all_tasks_done}"
        )
        self._queue.join()
        self._debug("queue join complete")

    def set_upstream(self, upstream: "PipelineSegment"):
        self._upstream = upstream

    def _debug(self, message: str):
        if self._debug_enabled:
            print(
                f"{datetime.datetime.now().strftime('%H:%M:%S')}: {self.__class__.__name__}/{threading.current_thread().name}: {message}",
                file=sys.stderr,
            )

    def thread_name(self, method_name="run"):
        self._thread_id += 1
        return f"{self.__class__.__name__}-{method_name}-{self._thread_id}"


class Source(PipelineSegment):
    """A pipeline segment that goes at the top. Only produces. Implement new_item_iterable."""

    def __init__(self):
        super().__init__()
        self._thread = None

    @abstractmethod
    def new_item_iterable(self) -> Iterable[Any]:
        pass

    def start(self):
        super().start()
        self._thread = Thread(target=self.run, name=self.thread_name(), daemon=True)
        self._thread.start()

    def run(self):
        self._debug("starting run")
        self._work_done.clear()
        try:
            for item in self.new_item_iterable():
                self._queue.put(item)
        except Exception as e:
            self._debug(f"exception {e} from iterable; ending early")
        self._work_done.set()
        self._debug(f"finished run")

    def join(self):
        super().join()
        self._thread.join()


class Pipe(PipelineSegment):
    """A pipeline segment that goes in the middle. Both consumes and produces. Implement handle_item."""

    def __init__(self, thread_count=1):
        super().__init__()
        self.thread_count = thread_count
        self._workers = []

    def start(self):
        self._work_done.clear()

        for i in range(self.thread_count):
            thread = Thread(target=self.run, name=self.thread_name())
            thread.start()
            self._workers.append(thread)
        Thread(target=self._notice_complete).start()

    @abstractmethod
    def handle_item(self, item) -> Optional[Any]:
        """
        Takes in an item from the previous stage, returns the item for the next stage. If for your use
        one input item produces multiple items, then don't return anything, instead calling
        self.downstream_put for each output item.
        """
        pass

    def _notice_complete(self):
        for worker in self._workers:
            self._debug(f"joining {worker}")
            worker.join()
        self._work_done.set()

    def run(self):
        self._debug(f"starting run")
        item = None
        while not self._upstream.done():
            self._debug(f"trying get")
            try:
                item = self.upstream_get()
                result = self.handle_item(item)
                if result:
                    self.downstream_put(result)
                self.completed += 1
                self.upstream_task_done()
                self._debug(f"success with {item} -> {result}")
            except queue.Empty:
                pass  # that's cool
                self._debug(f"empty")
            except Exception as e:
                self._debug(f"skipping item; exception {e} while processing {item}")
                self.upstream_task_done()

        self._debug(f"run finished")

    def join(self):
        self._debug(f"joining super")
        super().join()
        self._debug(f"joining threads")
        for thread in self._workers:
            thread.join()
        self._debug(f"join done")


class NullCache(dict):
    """Compatible with diskcache.Cache, but does nothing."""

    def __setitem__(self, __key, __value):
        pass

    def __enter__(self):
        return self

    def __exit__(self, __type, __value, __traceback):
        pass


class CachingPipe(Pipe):
    """A Pipe that optionally caches results the given directory. Implement key and handle_uncached_item."""

    def __init__(self, thread_count=1, cache_path=None):
        super().__init__(thread_count)

        if cache_path:
            self.cache = diskcache.Cache(cache_path).__enter__()
        else:
            self.cache = NullCache()

    def handle_item(self, item) -> Optional[Any]:
        cache_key = self.key(item)
        self._debug(f"looking for {cache_key} in cache")
        if cache_key in self.cache:
            self._debug(f"cache entry found")
            return self.cache[cache_key]
        else:
            self._debug(f"cache entry not found; processing and saving")
            result = self.handle_uncached_item(item)
            self.cache[cache_key] = result
            return result

    @abstractmethod
    def handle_uncached_item(self, item):
        """Do the work, returning the thing that you'd like to be cached and passed forward."""
        pass

    @abstractmethod
    def key(self, item):
        """The cache key for the pipeline item."""
        pass

    def join(self):
        super().join()
        self._debug(f"run complete with {self.cache} having {len(self.cache)} items ")
        self.cache.__exit__(None, None, None)


class Sink(PipelineSegment):
    """A pipeline segment that goes at the bottom. Doesn't produce, only consumes. Implement handle_item."""

    def __init__(self):
        super().__init__()
        self._thread = None

    def run(self):
        self._debug(f"run starting")

        self._work_done.clear()
        while not self._upstream.done():
            item = None
            try:
                item = self.upstream_get()
                self._debug(f"handling {item}")
                self.handle_item(item)
                self._debug(f"handled {item}")
                self.upstream_task_done()
                self.completed += 1
            except queue.Empty:
                # that's cool
                self._debug(f"get was empty")
            except Exception as e:
                self._debug(f"exception {e} handling {item}, skipping")
                self.upstream_task_done()

        self._work_done.set()
        self._debug(f"finished run with upstream done")

    def start(self):
        self._thread = Thread(target=self.run)
        self._thread.start()

    @abstractmethod
    def handle_item(self, item) -> None:
        """Receives a work item from the previous stage. No need to return anything here."""
        pass

    def join(self):
        super().join()
        self._debug(f"joining thread {self._thread}")
        self._thread.join()
        self._debug(f"thread join complete")


class Pipeline:
    def __init__(
        self,
        *segments: PipelineSegment,
        debug: bool = False,
        progress_callback: Optional[Callable] = None,
    ):
        super().__init__()
        self._segments = segments
        self.progress_callback = progress_callback

        self._debug_enabled = debug
        for s in self._segments:
            s._debug_enabled = debug

        assert isinstance(self.source, Source)
        assert isinstance(self.sink, Sink)

        for a, b in zip(segments[:-1], segments[1:]):
            b.set_upstream(a)

    @property
    def source(self):
        return self._segments[0]

    @property
    def sink(self):
        return self._segments[-1]

    def run(self):
        self._debug(f"pipeline run starting")

        self.report_progress()

        for segment in self._segments:
            segment.start()

        if self.progress_callback:
            while not self.sink.done():
                self.report_progress()

        for segment in self._segments:
            self._debug(f"joining {segment}")
            segment.join()
        self._debug(f"pipeline run complete")

        self.report_progress()

    def report_progress(self):
        if self.progress_callback:
            self.progress_callback({"completed": self.sink.completed})

    def _debug(self, message: str):
        if self._debug_enabled:
            print(
                f"{self.__class__.__name__}/{threading.current_thread().name}: {message}",
                file=sys.stderr,
            )
