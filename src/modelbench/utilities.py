import multiprocessing
from collections import defaultdict
from typing import List, Callable, Mapping, Sequence
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn, Task, Text

from modelbench.modelgauge_runner import ModelGaugeSut
from modelbench.benchmarks import BenchmarkDefinition


def group_by_key(items: Sequence, key=Callable) -> Mapping:
    """Group `items` by the value returned by `key(item)`."""
    groups = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    return groups


class StatusColumn(TaskProgressColumn):
    def render(self, task: Task) -> Text: 
        if ("error" in task.fields):
            return Text(text=" ⚠️ ", style="yellow on red")
        return super().render(task)


class ProgressBars():
    def __init__(self, benchmarks: List[BenchmarkDefinition], suts: List[ModelGaugeSut], debug=False):
        self._debug = debug
        self.manager = None

        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            StatusColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            refresh_per_second=1,
            speed_estimate_period=60 * 3,
        )

        # currently the progress bars are at a per test resolution
        # to provid more detail modelgauge would need to update as well
        self.num_tests_per_sut = 0
        for benchmark_definition in benchmarks:
            self.num_tests_per_sut += len(benchmark_definition.hazards())

        # create an overall progress bar for all the suts
        self.overallProgress = self.progress.add_task(
            "[green]Overall progress:",
            total=self.num_tests_per_sut * len(suts),
        )

        # create an individual progress bar for each sut
        self.progressBars = {}
        for sut in suts:
            self.progressBars[sut.name] = self.progress.add_task(
                sut.name,
                total=self.num_tests_per_sut,
            )

    def __enter__(self):
        self.progress.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        self.progress.__exit__(type, value, traceback)

        if (self.manager):
            self.manager.__exit__(type, value, traceback)

    def update_total_bar(self):
        self.progress.update(
            self.overallProgress,
            completed=sum([
                    self.num_tests_per_sut
                if "error" in self.progress._tasks[bar].fields
                else self.progress._tasks[bar].completed
                for bar in self.progressBars.values()
            ])
        )

    def handle_action(self, action):
        if (action["type"] == "finished_sut_test"):
            # add one to the sut progress bar
            self.progress.advance(self.progressBars[action["sut_name"]])
            self.update_total_bar()
        if (action["type"] == "debug"):
            # only print if in debug mode
            if (self._debug):
                self.progress.console.log(action["text"])
        if (action["type"] == "error"):
            # error in task
            name = action["sut_name"]
            self.progress.update(
                self.progressBars[name],
                error=True,
                finished=True,
                description=f"[red]{name}"
            )
            self.update_total_bar()

            # log error
            self.progress.console.log(action["text"])
            self.progress.console.log(action["error"])
            self.progress.console.log(action["trace"])

    def parallel_queue(self):
        if not self.manager:
            self.manager = multiprocessing.Manager()
            self.manager.__enter__()

        return self.manager.Queue()

    def sequential_queue(self):
        class SQueue:
            def __init__(self, logger):
                self.logger = logger

            def put(self, action):
                self.logger.handle_action(action)

        return SQueue(self)
