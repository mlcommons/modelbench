import dataclasses
import pathlib
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Iterable, Sequence, List

from modelgauge.annotation import Annotation
from modelgauge.annotator import CompletionAnnotator
from modelgauge.base_test import PromptResponseTest, TestResult
from modelgauge.dependency_helper import FromSourceDependencyHelper
from modelgauge.pipeline import Source, Pipe, Sink, Pipeline
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

from modelbench.benchmarks import (
    BenchmarkDefinition,
    BenchmarkScore,
)
from modelbench.suts import ModelGaugeSut


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

    def measure_quality(self, item: "BenchmarkPipelineItem"):
        annotations = SUTCompletionAnnotations(
            completion=item.sut_response.completions[0],
            annotations={k: Annotation.from_instance(v) for k, v in item.annotations.items()},
        )
        a = PromptInteractionAnnotations(
            prompt=item.test_item.prompts[0],
            response=SUTResponseAnnotations(completions=[annotations]),
        )
        item.add_measurement(
            self.actual_test.measure_quality(TestItemAnnotations(test_item=item.test_item, interactions=[a]))
        )

    def aggregate_measurements(self, items: List["BenchmarkPipelineItem"]):
        mtis = []
        for i in items:
            measurements = i.measurements
            print(f"measurements {measurements}")
            mti = MeasuredTestItem(test_item=i.test_item, measurements=measurements)

            mtis.append(mti)
        return self.actual_test.aggregate_measurements(mtis)


@dataclass
class BenchmarkPipelineItem:  # TODO: give a more domain-ish name?
    test: ModelgaugeTestWrapper
    test_item: TestItem  # TODO: maybe I don't need to carry all of the TestItem around?
    sut: ModelGaugeSut = None
    sut_response: SUTResponse = None
    annotations: dict[str, Annotation] = dataclasses.field(default_factory=dict)
    measurements = {}

    def prompt_with_context(self) -> PromptWithContext:
        return self.test_item.prompts[0]

    def completion(self) -> SUTCompletion:
        return self.sut_response.completions[0]

    def add_measurement(self, measurement: dict):
        self.measurements.update(measurement)


class BenchmarkRun:
    benchmark_scores: dict[BenchmarkDefinition, dict[ModelGaugeSut, BenchmarkScore]]
    benchmarks: Sequence[BenchmarkDefinition]

    def __init__(self, runner: "BenchmarkRunner"):
        super().__init__()
        # copy the starting state, mainly for later inspection
        self.pipeline_segments = []
        self.test_data_path = runner.data_dir / "tests"
        self.secrets = runner.secrets
        self.benchmarks = runner.benchmarks
        self.suts = runner.suts
        self.max_items = runner.max_items
        self.finished_items = defaultdict(lambda: defaultdict(lambda: list()))
        self.benchmark_scores = defaultdict(dict)

        self._test_lookup = {}
        for b in self.benchmarks:
            for h in b.hazards():
                for t in h.tests(self.secrets):
                    self._test_lookup[t] = ModelgaugeTestWrapper(t, self.test_data_path)

    def benchmark_scores(self) -> Mapping[BenchmarkDefinition, Mapping[ModelGaugeSut, BenchmarkScore]]:
        pass

    def add_finished_item(self, item: "BenchmarkPipelineItem"):
        self.finished_items[item.sut.key][item.test.uid].append(item)

    def items_for(self, sut, test):
        return self.finished_items[sut.key][test.uid]

    def tests(self):
        return self._test_lookup.values()


class BenchmarksSource(Source):

    def __init__(self, run: BenchmarkRun):
        super().__init__()
        self.benchmark_run = run

    def new_item_iterable(self) -> Iterable[BenchmarkPipelineItem]:
        for bm in self.benchmark_run.benchmarks:
            for h in bm.hazards():
                for t in h.tests(self.benchmark_run.secrets):
                    t = ModelgaugeTestWrapper(t, self.benchmark_run.test_data_path / "dependency_data")
                    items = t.make_test_items()
                    items = self.limit_to_max(items, self.benchmark_run.max_items)
                    for item in items:
                        pipeline_item = BenchmarkPipelineItem(t, item)
                        print(f"generated pipeline item {pipeline_item}", file=sys.stderr)
                        yield pipeline_item

    def limit_to_max(self, items: list, max_items: int):
        if max_items is not None:
            assert max_items > 0, f"invalid max_items: {max_items}"
            if max_items < len(items):
                rng = random.Random()
                rng.seed(0)
                rng.shuffle(items)
                return items[:max_items]
        return items


class BenchmarkSutAssigner(Pipe):
    def __init__(self, benchmark_run: BenchmarkRun):
        super().__init__()
        self.benchmark_run = benchmark_run

    def handle_item(self, item: BenchmarkPipelineItem):
        for sut in self.benchmark_run.suts:
            self.downstream_put(BenchmarkPipelineItem(item.test, item.test_item, sut))


class BenchmarkSutWorker(Pipe):

    def __init__(self, benchmark_run: BenchmarkRun, thread_count=1):
        super().__init__(thread_count)
        self.benchmark_run = benchmark_run

    def handle_item(self, item: BenchmarkPipelineItem) -> BenchmarkPipelineItem:
        # TODO: push some of this inside the ModelGaugeSut?
        mg_sut = item.sut.instance(self.benchmark_run.secrets)
        raw_request = mg_sut.translate_text_prompt(item.prompt_with_context().prompt)
        raw_response = mg_sut.evaluate(raw_request)
        response = mg_sut.translate_response(raw_request, raw_response)
        item.sut_response = response
        return item


class BenchmarkAnnotationWorker(Pipe):

    def __init__(self, benchmark_run: BenchmarkRun, thread_count=1):
        super().__init__(thread_count)
        self.benchmark_run = benchmark_run

    def handle_item(self, item: BenchmarkPipelineItem) -> BenchmarkPipelineItem:
        for annotator_key, annotator in item.test.get_annotators().items():
            annotator_request = annotator.translate_request(item.prompt_with_context(), item.completion())
            annotator_response = annotator.annotate(annotator_request)
            annotation = annotator.translate_response(annotator_request, annotator_response)
            item.annotations[annotator_key] = annotation
        item.test.measure_quality(item)
        return item


class BenchmarkResultsCollector(Sink):

    def __init__(self, benchmark_run: BenchmarkRun):
        super().__init__()
        self.benchmark_run = benchmark_run

    def handle_item(self, item) -> None:
        self.benchmark_run.add_finished_item(item)


class BenchmarkRunner:
    def __init__(self, data_dir: pathlib.Path):
        self.data_dir = data_dir
        self.secrets = None
        self.benchmarks = []
        self.suts = []
        self.annotators = {}
        self.max_items = 100

    def add_benchmark(self, benchmark: BenchmarkDefinition):
        self.benchmarks.append(benchmark)

    def add_sut(self, sut: ModelGaugeSut):
        self.suts.append(sut)

    def set_max_items(self, count: int):
        self.max_items = count

    def run(self) -> BenchmarkRun:
        run = BenchmarkRun(self)
        # preflight
        # build pipeline
        run.pipeline_segments.append(BenchmarksSource(run))
        run.pipeline_segments.append(BenchmarkSutAssigner(run))
        run.pipeline_segments.append(BenchmarkSutWorker(run))
        run.pipeline_segments.append(BenchmarkAnnotationWorker(run))
        run.pipeline_segments.append(BenchmarkResultsCollector(run))
        pipeline = Pipeline(
            *run.pipeline_segments,
            # progress_callback=progress_callback,
            debug=True,
        )

        # run pipeline
        pipeline.run()

        # gather results
        test_results = defaultdict(dict)
        for sut in run.suts:
            for test in run.tests():
                items = run.items_for(sut, test)
                test_result = test.aggregate_measurements(items)
                test_results[sut][test] = test_result

        # calculate scores
        for benchmark_definition in run.benchmarks:
            for sut in run.suts:
                hazard_scores = []
                for hazard in benchmark_definition.hazards():
                    sut_scores = {}
                    for test in hazard.tests(run.secrets):
                        test_result = test.aggregate_measurements(run.items_for(sut, test))
                        sut_scores[test.uid] = TestRecord(
                            test_uid=test.uid,
                            test_initialization=test.initialization_record,
                            dependency_versions=run._test_lookup[test].dependency_helper.versions_used(),
                            sut_uid=sut._instance.uid,
                            sut_initialization=sut._instance.initialization_record,
                            test_item_records=[],
                            result=TestResult.from_instance(test_result),
                        )

                        print(f"RESULT: {test_result}")
                    hazard.score(sut_scores)
                run.benchmark_scores[benchmark_definition][sut] = BenchmarkScore(
                    benchmark_definition, sut, hazard_scores, end_time=datetime.now()
                )

        return run

    def _score_benchmark(self, run, benchmark):
        for sut in self.suts:
            self._score_sut(run, benchmark, sut)
