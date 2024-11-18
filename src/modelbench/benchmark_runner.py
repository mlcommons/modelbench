import json
import logging
import pathlib
import random
import sys
import time
import traceback
from abc import abstractmethod
from collections import defaultdict
from datetime import datetime
from multiprocessing.pool import ThreadPool
from typing import Any, Iterable, Optional, Sequence

from modelgauge.annotator import CompletionAnnotator
from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.base_test import PromptResponseTest, TestResult
from modelgauge.config import raise_if_missing_from_config
from modelgauge.pipeline import NullCache, Pipe, Pipeline, Sink, Source
from modelgauge.prompt import TextPrompt
from modelgauge.records import TestRecord
from modelgauge.single_turn_prompt_response import PromptWithContext, TestItem
from modelgauge.sut import SUTCompletion, SUTResponse

from pydantic import BaseModel
from tqdm import tqdm

from modelbench.benchmark_runner_items import ModelgaugeTestWrapper, TestRunItem, Timer
from modelbench.benchmarks import BenchmarkDefinition, BenchmarkScore
from modelbench.cache import DiskCache, MBCache
from modelbench.run_journal import RunJournal
from modelbench.suts import ModelGaugeSut

logger = logging.getLogger(__name__)


class RunTracker:
    """
    A base class to encapsulate run tracking. Lets you limit update frequency to minimize output noise.
    To subclass, the minimum is implementing _on_update. If you want no output, just use the
    NullRunTracker.
    """

    def __init__(self, seconds_per_update: float = 1.0):
        super().__init__()
        self.seconds_per_update = seconds_per_update
        self.last_update = 0
        self.total_items = 0

    def start(self, total_items: int):
        self.total_items = total_items

    def update(self, finished_items: int):
        if self._now() > self.seconds_per_update + self.last_update:
            self._on_update(finished_items)
            self.last_update = self._now()

    def done(self):
        self._on_update(self.total_items)

    @abstractmethod
    def _on_update(self, finished_items: int):
        pass

    def _now(self):
        return time.time()


class NullRunTracker(RunTracker):
    def _on_update(self, finished_items: int):
        pass


class TqdmRunTracker(RunTracker):
    def start(self, total_items: int):
        super().start(total_items)
        self.pbar = tqdm(total=self.total_items, unit="items")
        self.previous_count = 0

    def _on_update(self, finished_items: int):
        self.pbar.update(finished_items - self.previous_count)
        self.previous_count = finished_items

    def done(self):
        super().done()
        self.pbar.close()


class JsonRunTracker(RunTracker):
    def start(self, total_items: int):
        super().start(total_items)
        self._on_update(0)

    def _on_update(self, finished_items: int):
        print(json.dumps({"progress": finished_items / self.total_items}), file=sys.stderr)


class TestRunBase:
    tests: list[ModelgaugeTestWrapper]

    def __init__(self, runner: "TestRunnerBase"):
        super().__init__()

        # copy the starting state
        self.pipeline_segments = []
        self.data_dir = runner.data_dir
        self.test_data_path = self.data_dir / "tests"
        self.secrets = runner.secrets
        self.suts = runner.suts
        self.max_items = runner.max_items
        self.tests = []
        self.test_annotators = {}
        self._test_lookup = {}
        self.run_tracker = runner.run_tracker
        self.completed_item_count = 0

        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.run_id = datetime.now().strftime("run-%Y%m%d-%H%M%S-%f")
        journal_dir = self.data_dir / "journals"
        journal_dir.mkdir(exist_ok=True, parents=True)
        self.journal = RunJournal(journal_dir / f"journal-{self.run_id}.jsonl.zst")

        self.caches = {}
        self.cache_starting_size = {}

        # set up for result collection
        self.finished_items = defaultdict(lambda: defaultdict(lambda: list()))
        self.failed_items = defaultdict(lambda: defaultdict(lambda: list()))
        self.test_records = defaultdict(dict)

    def add_test(self, test: PromptResponseTest):
        wrapped = ModelgaugeTestWrapper(test, self.test_data_path)
        self.tests.append(wrapped)
        self._test_lookup[test] = wrapped
        self._add_test_annotators(test)

    def _add_test_annotators(self, test: PromptResponseTest):
        # Check for missing secrets without instantiating any objects
        missing_secrets = []
        for annotator_uid in test.get_annotators():
            missing_secrets.extend(ANNOTATORS.get_missing_dependencies(annotator_uid, secrets=self.secrets))
        raise_if_missing_from_config(missing_secrets)

        annotators = []
        for annotator_uid in test.get_annotators():
            annotators.append(ANNOTATORS.make_instance(annotator_uid, secrets=self.secrets))
        self.test_annotators[test.uid] = annotators

    def add_finished_item(self, item: "TestRunItem"):
        if item.completion() and item.annotations and not item.exceptions:
            self.finished_items[item.sut.key][item.test.uid].append(item)
            self.journal.item_entry("item finished", item)
        else:
            self.failed_items[item.sut.key][item.test.uid].append(item)
            self.journal.item_entry(
                "item failed",
                item,
                completion=bool(item.completion()),
                annotations=len(item.annotations),
                fatal_exceptions=len(item.exceptions),
            )

        self.completed_item_count += 1

    def add_test_record(self, test_record: TestRecord):
        self.test_records[test_record.test_uid][test_record.sut_uid] = test_record

    def finished_items_for(self, sut, test) -> Sequence[TestItem]:
        return self.finished_items[sut.key][test.uid]

    def failed_items_for(self, sut, test) -> Sequence[TestItem]:
        return self.failed_items[sut.key][test.uid]

    def annotators_for_test(self, test: PromptResponseTest) -> Sequence[CompletionAnnotator]:
        return self.test_annotators[test.uid]

    def cache_for(self, cache_name: str):
        if self.data_dir:
            result = DiskCache(self.data_dir / cache_name)
        else:
            result = NullCache()

        self.caches[cache_name] = result
        self.cache_starting_size[cache_name] = len(result)
        return result

    def cache_info(self):
        result = []
        for key in self.caches.keys():
            result.append(f"  {key}: {self.caches[key]}")
            result.append(f"  {key}: started with {self.cache_starting_size[key]}")
            result.append(f"  {key}: finished with {len(self.caches[key])}")
        return "\n".join(result)


class TestRun(TestRunBase):
    tests: list[ModelgaugeTestWrapper]

    def __init__(self, runner: "TestRunner"):
        super().__init__(runner)
        # copy the starting state
        for test in runner.tests:
            self.add_test(test)


class BenchmarkRun(TestRunBase):
    benchmark_scores: dict[BenchmarkDefinition, dict[ModelGaugeSut, BenchmarkScore]]
    benchmarks: Sequence[BenchmarkDefinition]

    def __init__(self, runner: "BenchmarkRunner"):
        super().__init__(runner)
        self.benchmarks = runner.benchmarks
        self.benchmark_scores = defaultdict(dict)

        for b in self.benchmarks:
            for h in b.hazards():
                for t in h.tests(self.secrets):
                    self.add_test(t)


class IntermediateCachingPipe(Pipe):
    """
    Unlike CachingPipe, which caches the final result of this stage,
    this just makes a cache available for internal use to cache intermediate results.
    """

    def __init__(self, cache: MBCache, thread_count=1):
        super().__init__(thread_count)
        self.cache = cache

    def handle_item(self, item) -> Optional[Any]:
        pass

    def join(self):
        super().join()
        self.cache.__exit__(None, None, None)


class TestRunItemSource(Source):
    def __init__(self, run: TestRunBase, queue_maxsize=0):
        super().__init__(queue_maxsize)
        self.test_run = run

    def new_item_iterable(self, quiet=False) -> Iterable[TestRunItem]:
        for t in self.test_run.tests:
            all_items = t.make_test_items()
            items = self.limit_to_max(all_items, self.test_run.max_items)
            if not quiet:
                self.test_run.journal.raw_entry("using test items", using=len(items), total=len(all_items), test=t.uid)
            for item in items:
                yield TestRunItem(t, item)

    def limit_to_max(self, items: list, max_items: int):
        if max_items is not None:
            assert max_items > 0, f"invalid max_items: {max_items}"
            if max_items < len(items):
                rng = random.Random()
                rng.seed(0)
                rng.shuffle(items)
                return items[:max_items]
        return items


class TestRunSutAssigner(Pipe):
    def __init__(self, test_run: TestRunBase):
        super().__init__()
        self.test_run = test_run

    def handle_item(self, item: TestRunItem):
        for sut in self.test_run.suts:
            run_item = TestRunItem(item.test, item.test_item, sut)
            self.test_run.journal.item_entry(
                "queuing item",
                run_item,
                prompt_text=item.prompt_with_context().prompt.text,
            )

            self.downstream_put(run_item)


class TestRunSutWorker(IntermediateCachingPipe):
    def __init__(self, test_run: TestRunBase, cache: MBCache, thread_count=1):

        super().__init__(cache, thread_count)
        self.test_run = test_run

    def handle_item(self, item: TestRunItem):
        mg_sut = item.sut.instance(self.test_run.secrets)
        raw_request = mg_sut.translate_text_prompt(item.prompt_with_context().prompt)
        cache_key = raw_request.model_dump_json(exclude_none=True)
        self._debug(f"looking for {cache_key} in cache")
        try:
            if cache_key in self.cache:
                self._debug(f"cache entry found")
                raw_response = self.cache[cache_key]
                self.test_run.journal.item_entry("using cached sut response", item, response=raw_response)

            else:
                self._debug(f"cache entry not found; processing and saving")
                with Timer() as timer:
                    raw_response = mg_sut.evaluate(raw_request)
                self.cache[cache_key] = raw_response
                self.test_run.journal.item_entry(
                    "fetched sut response", item, run_time=timer, request=raw_request, response=raw_response
                )

            response = mg_sut.translate_response(raw_request, raw_response)
            item.sut_response = response
            self.test_run.journal.item_entry("translated sut response", item, response=response)

        except Exception as e:
            item.exceptions.append(e)
            self.test_run.journal.item_exception_entry("sut exception", item, e)
            logger.error(f"failure handling sut item {item}:", exc_info=e)
        return item


class TestRunAnnotationWorker(IntermediateCachingPipe):
    def __init__(self, test_run: TestRunBase, cache: MBCache, thread_count=1, cache_path=None):
        super().__init__(cache, thread_count)
        self.test_run = test_run

    def handle_item(self, item: TestRunItem) -> TestRunItem:
        try:
            if item.completion():
                with Timer() as timer:
                    self.collect_annotations(item)
                    item.test.measure_quality(item)
                self.test_run.journal.item_entry(
                    "measured item quality", item, measurements=item.measurements, run_time=timer
                )
        except Exception as e:
            item.exceptions.append(e)
            logger.error(f"failure handling annnotation for {item}", exc_info=e)
            self.test_run.journal.item_exception_entry("annotation exception", item, e)
        return item

    def collect_annotations(self, item):
        for annotator in self.test_run.annotators_for_test(item.test):
            try:
                annotator_request = annotator.translate_request(item.prompt_with_context(), item.completion())
                cache_key = self.make_cache_key(annotator_request)
                self._debug(f"looking for {cache_key} in cache")
                if cache_key in self.cache:
                    self._debug(f"cache entry found")
                    annotator_response = self.cache[cache_key]
                    self.test_run.journal.item_entry(
                        "using cached annotator response",
                        item,
                        annotator=annotator.uid,
                        response=annotator_response,
                    )
                else:
                    self._debug(f"cache entry not found; processing and saving")
                    with Timer() as timer:
                        annotator_response = annotator.annotate(annotator_request)
                    self.cache[cache_key] = annotator_response
                    self.test_run.journal.item_entry(
                        "fetched annotator response",
                        item,
                        annotator=annotator.uid,
                        run_time=timer,
                        response=annotator_response,
                    )

                annotation = annotator.translate_response(annotator_request, annotator_response)
                self.test_run.journal.item_entry(
                    "translated annotation", item, annotator=annotator.uid, annotation=annotation
                )

                item.annotations[annotator.uid] = annotation
            except Exception as e:
                item.exceptions.append(e)
                logger.error(f"failure handling annotation for {annotator.uid} and {item}", exc_info=e)
                self.test_run.journal.item_exception_entry("annotator exception", item, e, annotator=annotator.uid)

    def make_cache_key(self, annotator_request):
        if isinstance(annotator_request, BaseModel):
            return annotator_request.model_dump_json(exclude_none=True)
        elif isinstance(annotator_request, str):
            return annotator_request
        else:
            raise ValueError(f"Don't know how to make a key out of {annotator_request.__class__}: {annotator_request}")


class TestRunResultsCollector(Sink):
    def __init__(self, test_run: TestRunBase):
        super().__init__()
        self.test_run = test_run

    def handle_item(self, item) -> None:
        self.test_run.add_finished_item(item)
        self.test_run.run_tracker.update(self.test_run.completed_item_count)


class TestRunnerBase:
    def __init__(self, data_dir: pathlib.Path):
        self.debug = False
        self.data_dir = data_dir
        self.secrets = None
        self.suts = []
        self.max_items = 10
        self.thread_count = 1
        self.run_tracker = NullRunTracker()

    def add_sut(self, sut: ModelGaugeSut):
        self.suts.append(sut)

    def _check_ready_to_run(self):
        if not self.secrets:
            raise ValueError("must set secrets")

        if not self.suts:
            raise ValueError("must call add_sut() at least once")

    def _check_external_services(self, run: TestRunBase):
        self._check_suts_working(run)
        self._check_annotators_working(run)

    def _check_suts_working(self, run: TestRunBase):
        def check_sut(sut: ModelGaugeSut):
            try:
                mg_sut = sut.instance(self.secrets)
                raw_request = mg_sut.translate_text_prompt(TextPrompt(text="Why did the chicken cross the road?"))
                raw_response = mg_sut.evaluate(raw_request)
                response: SUTResponse = mg_sut.translate_response(raw_request, raw_response)
                return bool(response.completions)
            except Exception as e:
                logger.error(f"initial check failure for {sut}", exc_info=e)
                print(f"initial check failure for {sut}")
                traceback.print_exc()

                return False

        with ThreadPool(len(run.suts)) as pool:
            suts_worked = pool.map(check_sut, self.suts)
            if not all(suts_worked):
                raise RuntimeError(
                    f"Not all SUTs are ready to go. Status: {dict(zip([s.uid for s in self.suts], suts_worked))}"
                )

    def _check_annotators_working(self, run: TestRunBase):
        def check_annotator(annotator: CompletionAnnotator):
            try:
                raw_request = annotator.translate_request(
                    PromptWithContext(
                        prompt=TextPrompt(text="Why did the chicken cross the road?"), source_id="ignored"
                    ),
                    SUTCompletion(text="To get to the other side."),
                )
                raw_response = annotator.annotate(raw_request)
                response = annotator.translate_response(raw_request, raw_response)
                return bool(response)

            except Exception as e:
                logger.error(f"initial check failure for {annotator}", exc_info=e)
                print(f"initial check failure for {annotator}")
                traceback.print_exc()
                return False

        annotators = set(a for l in run.test_annotators.values() for a in l)
        with ThreadPool(len(annotators)) as pool:
            annotators_worked = pool.map(check_annotator, annotators)
            if not all(annotators_worked):
                raise RuntimeError(
                    f"Not all annotators are ready to go. Status: {dict(zip([a.uid for a in annotators], annotators_worked))}"
                )

    def _calculate_test_results(self, test_run):
        for sut in test_run.suts:
            for test in test_run.tests:
                finished_items = test_run.finished_items_for(sut, test)
                test_result = test.aggregate_measurements(finished_items)
                test_record = self._make_test_record(test_run, sut, test, test_result)
                test_run.add_test_record(test_record)
                test_run.journal.raw_entry(
                    "test scored", sut=sut.uid, test=test.uid, items_finished=len(finished_items), result=test_result
                )

    def _make_test_record(self, run, sut, test, test_result):
        return TestRecord(
            test_uid=test.uid,
            test_initialization=test.initialization_record,
            dependency_versions=test.dependency_helper.versions_used(),
            sut_uid=sut._instance.uid,
            sut_initialization=sut._instance.initialization_record,
            test_item_records=[],
            test_item_exceptions=[],
            result=TestResult.from_instance(test_result),
        )

    def _build_pipeline(self, run):
        run.pipeline_segments.append(TestRunItemSource(run, queue_maxsize=self.thread_count * 4))
        run.pipeline_segments.append(TestRunSutAssigner(run))
        run.pipeline_segments.append(TestRunSutWorker(run, run.cache_for("sut_cache"), thread_count=self.thread_count))
        run.pipeline_segments.append(
            TestRunAnnotationWorker(run, run.cache_for("annotator_cache"), thread_count=self.thread_count)
        )
        run.pipeline_segments.append(TestRunResultsCollector(run))
        pipeline = Pipeline(
            *run.pipeline_segments,
            debug=self.debug,
        )
        return pipeline

    def _expected_item_count(self, the_run: TestRunBase, pipeline: Pipeline):
        return len(the_run.suts) * len(list(pipeline.source.new_item_iterable(quiet=True)))


class TestRunner(TestRunnerBase):
    def __init__(self, data_dir: pathlib.Path):
        super().__init__(data_dir)
        self.tests = []

    def add_test(self, test: PromptResponseTest):
        self.tests.append(test)

    def _check_ready_to_run(self):
        super()._check_ready_to_run()
        if not self.tests:
            raise ValueError("must call add_test() at least once")

    def run(self) -> TestRun:
        self._check_ready_to_run()
        test_run = TestRun(self)
        pipeline = self._build_pipeline(test_run)
        test_run.run_tracker.start(self._expected_item_count(test_run, pipeline))

        pipeline.run()

        self._calculate_test_results(test_run)
        test_run.run_tracker.done()
        return test_run


class BenchmarkRunner(TestRunnerBase):
    def __init__(self, data_dir: pathlib.Path):
        super().__init__(data_dir)
        self.benchmarks = []

    def add_benchmark(self, benchmark: BenchmarkDefinition):
        self.benchmarks.append(benchmark)

    def _check_ready_to_run(self):
        super()._check_ready_to_run()
        if not self.benchmarks:
            raise ValueError("must call add_benchmark() at least once")

    def run(self) -> BenchmarkRun:
        self._check_ready_to_run()

        benchmark_run = BenchmarkRun(self)
        self._check_external_services(benchmark_run)
        benchmark_run.journal.raw_entry(
            "starting run",
            run_id=benchmark_run.run_id,
            benchmarks=[b.uid for b in benchmark_run.benchmarks],
            tests=[t.uid for t in benchmark_run.tests],
            suts=[s.uid for s in benchmark_run.suts],
            max_items=benchmark_run.max_items,
            thread_count=self.thread_count,
        )
        for test in benchmark_run.tests:
            benchmark_run.journal.raw_entry(
                "test info",
                test=test.uid,
                initialization=test.initialization_record,
                sut_options=test.sut_options(),
                dependencies=test.dependencies(),
            )
        pipeline = self._build_pipeline(benchmark_run)
        benchmark_run.run_tracker.start(self._expected_item_count(benchmark_run, pipeline))
        benchmark_run.journal.raw_entry("running pipeline")
        with Timer() as timer:
            pipeline.run()

        total_items_finished = 0
        finished_item_counts = defaultdict(dict)
        for k1, d1 in benchmark_run.finished_items.items():
            for k2, l1 in d1.items():
                total_items_finished += len(d1)
                finished_item_counts[k1][k2] = len(d1)

        benchmark_run.journal.raw_entry(
            "finished pipeline",
            time=timer.elapsed,
            total_finished=total_items_finished,
            finished_counts=finished_item_counts,
        )

        self._calculate_test_results(benchmark_run)
        self._calculate_benchmark_scores(benchmark_run)
        benchmark_run.run_tracker.done()
        benchmark_run.journal.raw_entry("finished run", run_id=benchmark_run.run_id)
        for key, cache in benchmark_run.caches.items():
            cache = benchmark_run.caches[key]
            benchmark_run.journal.raw_entry(
                "cache info",
                type=key,
                cache=str(cache),
                start_count=benchmark_run.cache_starting_size[key],
                end_count=len(cache),
            )

        benchmark_run.journal.close()
        return benchmark_run

    def _calculate_benchmark_scores(self, benchmark_run):
        for benchmark_definition in benchmark_run.benchmarks:
            for sut in benchmark_run.suts:
                hazard_scores = []
                for hazard in benchmark_definition.hazards():
                    test_records = {}
                    for test in hazard.tests(benchmark_run.secrets):
                        records = benchmark_run.test_records[test.uid][sut.uid]
                        assert records, f"No records found for {benchmark_definition} {sut} {hazard} {test.uid}"
                        test_records[test.uid] = records

                    assert test_records, f"No records found for {benchmark_definition} {sut} {hazard}"

                    hazard_score = hazard.score(test_records)
                    hazard_scores.append(hazard_score)  # TODO: score needs way less
                    benchmark_run.journal.raw_entry(
                        "hazard scored",
                        benchmark=benchmark_definition.uid,
                        sut=sut.uid,
                        hazard=hazard.uid,
                        score=hazard_score.score.estimate,
                        reference=hazard.reference_standard(),
                        samples=hazard_score.score.samples,
                        numeric_grade=hazard_score.numeric_grade(),
                        text_grade=hazard_score.text_grade(),
                    )

                benchmark_score = BenchmarkScore(benchmark_definition, sut, hazard_scores, end_time=datetime.now())
                benchmark_run.benchmark_scores[benchmark_definition][sut] = benchmark_score
                benchmark_run.journal.raw_entry(
                    "benchmark scored",
                    benchmark=benchmark_definition.uid,
                    sut=sut.uid,
                    numeric_grade=benchmark_score.numeric_grade(),
                    text_grade=benchmark_score.text_grade(),
                    scoring_log=benchmark_score._scoring_log,
                )
