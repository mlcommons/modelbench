import dataclasses
import json
import pathlib
import random
import sys
import time
import traceback
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Iterable, Sequence, List, Optional, Any

import diskcache
from modelgauge.annotation import Annotation
from modelgauge.annotator import CompletionAnnotator
from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.base_test import PromptResponseTest, TestResult
from modelgauge.config import raise_if_missing_from_config
from modelgauge.dependency_helper import FromSourceDependencyHelper
from modelgauge.pipeline import Source, Pipe, Sink, Pipeline, NullCache
from modelgauge.records import TestRecord
from modelgauge.single_turn_prompt_response import (
    TestItem,
    PromptWithContext,
    MeasuredTestItem,
    TestItemAnnotations,
    PromptInteractionAnnotations,
    SUTResponseAnnotations,
    SUTCompletionAnnotations,
)
from modelgauge.sut import SUTResponse, SUTCompletion
from tqdm import tqdm

from modelbench.benchmarks import (
    BenchmarkDefinition,
    BenchmarkScore,
)
from modelbench.suts import ModelGaugeSut


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


class ModelgaugeTestWrapper:
    """An attempt at cleaning up the test interface"""

    def __init__(self, actual_test: PromptResponseTest, dependency_data_path):
        super().__init__()
        self.actual_test = actual_test
        self.uid = actual_test.uid
        self.dependency_data_path = dependency_data_path
        self.dependency_helper = FromSourceDependencyHelper(
            self.dependency_data_path, self.actual_test.get_dependencies(), required_versions={}
        )

    def make_test_items(self):
        return self.actual_test.make_test_items(self.dependency_helper)

    def __hash__(self):
        return self.uid.__hash__()

    def get_annotators(self) -> Mapping[str, CompletionAnnotator]:
        return self.actual_test.get_annotators()

    def measure_quality(self, item: "TestRunItem"):
        annotations = SUTCompletionAnnotations(
            completion=item.sut_response.completions[0],
            annotations={k: Annotation.from_instance(v) for k, v in item.annotations.items()},
        )
        a = PromptInteractionAnnotations(
            prompt=item.test_item.prompts[0],
            response=SUTResponseAnnotations(completions=[annotations]),
        )
        measurement = self.actual_test.measure_quality(TestItemAnnotations(test_item=item.test_item, interactions=[a]))
        item.add_measurement(measurement)

    def aggregate_measurements(self, items: List["TestRunItem"]):
        mtis = []
        for i in items:
            mti = MeasuredTestItem(test_item=i.test_item, measurements=i.measurements)
            mtis.append(mti)
        return self.actual_test.aggregate_measurements(mtis)

    @property
    def initialization_record(self):
        return self.actual_test.initialization_record


@dataclass
class TestRunItem:
    """The data related to running a single test item"""

    test: ModelgaugeTestWrapper
    test_item: TestItem
    sut: ModelGaugeSut = None
    sut_response: SUTResponse = None
    annotations: dict[str, Annotation] = dataclasses.field(default_factory=dict)
    measurements: dict[str, float] = dataclasses.field(default_factory=dict)
    exception = None

    def prompt_with_context(self) -> PromptWithContext:
        return self.test_item.prompts[0]

    def completion(self) -> SUTCompletion:
        if self.sut_response and self.sut_response.completions:
            return self.sut_response.completions[0]

    def add_measurement(self, measurement: dict):
        self.measurements.update(measurement)


class TestRunBase:
    def __init__(self, runner: "TestRunnerBase"):
        super().__init__()
        # copy the starting state
        self.pipeline_segments = []
        self.test_data_path = runner.data_dir / "tests"
        self.secrets = runner.secrets
        self.suts = runner.suts
        self.max_items = runner.max_items
        self.tests = []
        self.test_annotators = {}
        self._test_lookup = {}
        self.run_tracker = runner.run_tracker
        self.completed_item_count = 0

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
        if item.completion() and item.annotations and not item.exception:
            self.finished_items[item.sut.key][item.test.uid].append(item)
        else:
            self.failed_items[item.sut.key][item.test.uid].append(item)
        self.completed_item_count += 1

    def add_test_record(self, test_record: TestRecord):
        self.test_records[test_record.test_uid][test_record.sut_uid] = test_record

    def finished_items_for(self, sut, test) -> Sequence[TestItem]:
        return self.finished_items[sut.key][test.uid]

    def failed_items_for(self, sut, test) -> Sequence[TestItem]:
        return self.failed_items[sut.key][test.uid]

    def annotators_for_test(self, test: PromptResponseTest) -> Sequence[CompletionAnnotator]:
        return self.test_annotators[test.uid]


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

    def __init__(self, thread_count=1, cache_path=None):
        super().__init__(thread_count)

        if cache_path:
            self.cache = diskcache.Cache(cache_path).__enter__()
        else:
            self.cache = NullCache()

    def handle_item(self, item) -> Optional[Any]:
        pass

    def join(self):
        super().join()
        self.cache.__exit__(None, None, None)


class TestRunItemSource(Source):

    def __init__(self, run: TestRunBase):
        super().__init__()
        self.test_run = run

    def new_item_iterable(self) -> Iterable[TestRunItem]:
        for t in self.test_run.tests:
            items = t.make_test_items()
            items = self.limit_to_max(items, self.test_run.max_items)
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
            self.downstream_put(TestRunItem(item.test, item.test_item, sut))


class TestRunSutWorker(IntermediateCachingPipe):

    def __init__(self, test_run: TestRunBase, thread_count=1, cache_path=None):
        super().__init__(thread_count, cache_path=cache_path)
        self.test_run = test_run

    def handle_item(self, item):
        mg_sut = item.sut.instance(self.test_run.secrets)
        raw_request = mg_sut.translate_text_prompt(item.prompt_with_context().prompt)
        cache_key = raw_request.model_dump_json(exclude_none=True)
        self._debug(f"looking for {cache_key} in cache")
        try:
            if cache_key in self.cache:
                self._debug(f"cache entry found")
                raw_response = self.cache[cache_key]
            else:
                self._debug(f"cache entry not found; processing and saving")
                raw_response = mg_sut.evaluate(raw_request)
                self.cache[cache_key] = raw_response

            response = mg_sut.translate_response(raw_request, raw_response)
            item.sut_response = response
        except Exception as e:
            item.exception = e
            print(traceback.format_exc(), file=sys.stderr)
        return item


class TestRunAnnotationWorker(IntermediateCachingPipe):

    def __init__(self, test_run: TestRunBase, thread_count=1, cache_path=None):
        super().__init__(thread_count, cache_path=cache_path)
        self.test_run = test_run

    def handle_item(self, item: TestRunItem) -> TestRunItem:
        try:
            if item.completion():
                self.collect_annotations(item)
        except Exception as e:
            item.exception = e
            print(traceback.format_exc(), file=sys.stderr)

        return item

    def collect_annotations(self, item):
        for annotator in self.test_run.annotators_for_test(item.test):
            annotator_request = annotator.translate_request(item.prompt_with_context(), item.completion())
            cache_key = annotator_request.model_dump_json(exclude_none=True)
            self._debug(f"looking for {cache_key} in cache")
            if cache_key in self.cache:
                self._debug(f"cache entry found")
                annotator_response = self.cache[cache_key]
            else:
                self._debug(f"cache entry not found; processing and saving")
                annotator_response = annotator.annotate(annotator_request)
                self.cache[cache_key] = annotator_response

            annotation = annotator.translate_response(annotator_request, annotator_response)
            item.annotations[annotator.uid] = annotation
        item.test.measure_quality(item)


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

    def _calculate_test_results(self, test_run):
        for sut in test_run.suts:
            for test in test_run.tests:
                test_result = test.aggregate_measurements(test_run.finished_items_for(sut, test))
                test_record = self._make_test_record(test_run, sut, test, test_result)
                test_run.add_test_record(test_record)

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
        run.pipeline_segments.append(TestRunItemSource(run))
        run.pipeline_segments.append(TestRunSutAssigner(run))
        run.pipeline_segments.append(
            TestRunSutWorker(run, thread_count=self.thread_count, cache_path=self.data_dir / "sut_cache")
        )
        run.pipeline_segments.append(
            TestRunAnnotationWorker(run, thread_count=self.thread_count, cache_path=self.data_dir / "annotator_cache")
        )
        run.pipeline_segments.append(TestRunResultsCollector(run))
        pipeline = Pipeline(
            *run.pipeline_segments,
            # progress_callback=progress_callback,
            debug=self.debug,
        )
        return pipeline

    def _expected_item_count(self, the_run: TestRunBase, pipeline: Pipeline):
        return len(the_run.suts) * len(list(pipeline.source.new_item_iterable()))


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
        pipeline = self._build_pipeline(benchmark_run)
        benchmark_run.run_tracker.start(self._expected_item_count(benchmark_run, pipeline))
        pipeline.run()

        self._calculate_test_results(benchmark_run)
        self._calculate_benchmark_scores(benchmark_run)
        benchmark_run.run_tracker.done()
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

                    hazard_scores.append(hazard.score(test_records))  # TODO: score needs way less
                benchmark_run.benchmark_scores[benchmark_definition][sut] = BenchmarkScore(
                    benchmark_definition, sut, hazard_scores, end_time=datetime.now()
                )
