from typing import Dict
from unittest.mock import MagicMock

import pytest
from modelgauge.annotator import Annotator
from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.annotators.demo_annotator import DemoYBadAnnotation
from modelgauge.dependency_helper import DependencyHelper
from modelgauge.external_data import ExternalData
from modelgauge.load_plugins import load_plugins
from modelgauge.prompt import TextPrompt
from modelgauge.record_init import InitializationRecord
from modelgauge.secret_values import RawSecrets, get_all_secrets
from modelgauge.tests.safe import PersonaResult

from modelbench.benchmark_runner import *
from modelbench.hazards import SafeNvcHazard, HazardDefinition, HazardScore
from modelbench.scoring import ValueEstimate
from modelbench.suts import ModelGaugeSut


def fake_all_secrets(value="some-value") -> RawSecrets:
    secrets = get_all_secrets()
    raw_secrets: Dict[str, Dict[str, str]] = {}
    for secret in secrets:
        if secret.scope not in raw_secrets:
            raw_secrets[secret.scope] = {}
        raw_secrets[secret.scope][secret.key] = value
    return raw_secrets


class TestRunner:
    @pytest.fixture(scope="class", autouse=True)
    def load_plugins(self):
        load_plugins()

    @pytest.fixture()
    def item_from_test(self):
        return TestItem(prompts=[PromptWithContext(prompt=TextPrompt(text="Hello!"), source_id="hello")])

    @pytest.fixture()
    def a_test(self, item_from_test, fake_secrets):
        class ATest(PromptResponseTest):

            def __init__(self, uid: str):
                super().__init__(uid)
                self.initialization_record = InitializationRecord(
                    module="testing", class_name="a_test", args=[], kwargs={}
                )

            def get_dependencies(self) -> Mapping[str, ExternalData]:
                pass

            def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
                return [item_from_test]

            def get_annotators(self) -> Mapping[str, Annotator]:
                return {"demo_annotator": ANNOTATORS.make_instance("demo_annotator", secrets=fake_secrets)}

            def measure_quality(self, item: TestItemAnnotations) -> Dict[str, float]:
                ann = item.interactions[0].response.completions[0].get_annotation("demo_annotator", DemoYBadAnnotation)
                print(f"ann: {ann}")
                return {"badness": float(ann.badness)}

            def aggregate_measurements(self, items: List[MeasuredTestItem]):
                total_badness = 0
                badness_count = 0
                for item in items:
                    total_badness += item.measurements["badness"]
                    badness_count += 1
                return {"total_badness": total_badness, "badness_count": badness_count}

        return ATest("a_test")

    @pytest.fixture()
    def a_wrapped_test(self, a_test, tmp_path):
        return ModelgaugeTestWrapper(a_test, tmp_path)

    @pytest.fixture()
    def benchmark(self, a_test):
        class AHazard(HazardDefinition):
            def tests(self, secrets: RawSecrets) -> List[PromptResponseTest]:
                return [a_test]

            def score(self, sut_scores: Mapping[str, TestRecord]) -> HazardScore:
                count = 0
                total = 0
                for key, value in sut_scores.items():
                    d = value.result.data
                    count += d["badness_count"]
                    total += d["total_badness"]

                score = ValueEstimate.make(total / count, count)

                test_scores = {}
                return HazardScore(hazard_definition=self, score=score, test_scores=test_scores)

        class ABenchmark(BenchmarkDefinition):
            def _make_hazards(self) -> Sequence[HazardDefinition]:
                return [AHazard()]

        return ABenchmark()

    def a_run(self, tmp_path, **kwargs) -> BenchmarkRun:
        runner = BenchmarkRunner(tmp_path / "run")
        for key, value in kwargs.items():
            runner.__dict__[key] = value
        if runner.secrets is None:
            runner.secrets = fake_all_secrets()
        return BenchmarkRun(runner)

    @pytest.fixture()
    def sut_response(self):
        return SUTResponse(completions=[SUTCompletion(text="Hello, is it me you're looking for?")])

    @pytest.fixture()
    def hazard(self):
        pass

    def test_benchmark_source(self, fake_secrets, tmp_path, benchmark):
        bsa = BenchmarksSource(self.a_run(tmp_path, secrets=fake_secrets, max_items=1, benchmarks=[benchmark]))
        iterator = iter(bsa.new_item_iterable())
        first_item = next(iterator)
        print(first_item)
        assert isinstance(first_item, BenchmarkPipelineItem)
        assert isinstance(first_item.test_item, TestItem)
        with pytest.raises(StopIteration):
            print(next(iterator))

    def test_benchmark_sut_assigner(self, a_wrapped_test, tmp_path):
        sut_one = ModelGaugeSut("one")
        sut_two = ModelGaugeSut("two")
        test_item = TestItem(prompts=[])

        bsa = BenchmarkSutAssigner(self.a_run(tmp_path, suts=[sut_one, sut_two]))
        bsa.handle_item(BenchmarkPipelineItem(a_wrapped_test, test_item))

        assert bsa._queue.qsize() == 2
        item_one = bsa._queue.get()
        assert item_one.test_item == test_item
        assert item_one.sut == sut_one
        item_two = bsa._queue.get()
        assert item_two.test_item == test_item
        assert item_two.sut == sut_two

    def test_benchmark_sut_worker(self, item_from_test, a_wrapped_test, tmp_path):
        sut = ModelGaugeSut("demo_yes_no")
        run = self.a_run(tmp_path, suts=[sut])
        bsw = BenchmarkSutWorker(run)
        result = bsw.handle_item(BenchmarkPipelineItem(a_wrapped_test, item_from_test, sut))
        assert result.test_item == item_from_test
        assert result.sut == sut
        assert isinstance(result.sut_response, SUTResponse)
        assert result.sut_response.completions[0].text == "No"

    def test_benchmark_annotation_worker(self, a_wrapped_test, tmp_path, item_from_test, sut_response):
        sut = ModelGaugeSut("demo_yes_no")
        run = self.a_run(tmp_path, suts=[sut])
        baw = BenchmarkAnnotationWorker(run)
        pipeline_item = BenchmarkPipelineItem(
            a_wrapped_test, item_from_test, ModelGaugeSut("demo_yes_no"), sut_response
        )
        result = baw.handle_item(pipeline_item)
        print(result.annotations)
        assert result.annotations["demo_annotator"].badness == 1.0

    def test_benchmark_results_collector(self, tmp_path, a_wrapped_test, item_from_test, sut_response):
        run = self.a_run(tmp_path)
        brc = BenchmarkResultsCollector(run)
        sut = ModelGaugeSut("demo_yes_no")
        item = BenchmarkPipelineItem(a_wrapped_test, item_from_test, sut, sut_response)
        brc.handle_item(item)
        items = run.items_for(sut, a_wrapped_test)
        assert items == [item]

    def test_basic_run(self, tmp_path, fake_secrets, benchmark):
        runner = BenchmarkRunner(tmp_path)
        runner.secrets = fake_secrets

        runner.add_benchmark(benchmark)
        sut = ModelGaugeSut("demo_yes_no")
        runner.add_sut(sut)
        runner.set_max_items(1)
        run_result = runner.run()

        assert run_result.benchmark_scores
        assert run_result.benchmark_scores[benchmark][sut]

    def test_serialization_scratch(self, item_from_test, sut_response):
        annotator = ANNOTATORS.make_instance("demo_annotator", secrets=[])
        areq = annotator.translate_request(item_from_test.prompts[0], sut_response.completions[0])
        print(f"areq {areq}")
        aresp = annotator.annotate(areq)
        print(f"aresp {aresp}")
        resp = annotator.translate_response(areq, aresp)
        print(f"resp {resp.model_dump_json()}")
        ai = Annotation.from_instance(resp)
        print(f"ai {ai.model_dump_json()}")

    def test_scoring_scratch(self):
        h = SafeNvcHazard()
        safe_test_result = MagicMock()
        safe_test_result.persona_results = {"typical": PersonaResult(frac_safe=0.5, num_items=10)}
        test_record = MagicMock()
        test_record.result.to_instance = lambda _: safe_test_result
        s = h.score({"foo": test_record})
        print(s)

    # TODO: runner should blow up if not enough state is set
    # TODO: add caching
    # TODO: add stats
    # TODO: track errors
    # TODO: handle logs
