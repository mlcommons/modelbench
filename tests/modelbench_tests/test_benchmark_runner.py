import inspect
from typing import Dict
from unittest.mock import MagicMock

import pytest
from modelgauge.annotator import Annotator
from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.annotators.demo_annotator import DemoYBadAnnotation, DemoYBadAnnotator
from modelgauge.annotators.llama_guard_annotator import LlamaGuardAnnotation
from modelgauge.dependency_helper import DependencyHelper
from modelgauge.external_data import ExternalData
from modelgauge.load_plugins import load_plugins
from modelgauge.prompt import TextPrompt
from modelgauge.record_init import InitializationRecord
from modelgauge.secret_values import RawSecrets, get_all_secrets
from modelgauge.suts.together_client import TogetherChatRequest, TogetherChatResponse

from modelbench.benchmark_runner import *
from modelbench.hazards import HazardDefinition, HazardScore
from modelbench.scoring import ValueEstimate
from modelbench.suts import ModelGaugeSut

# fix pytest autodiscovery issue; see https://github.com/pytest-dev/pytest/issues/12749
for a_class in [i[1] for i in (globals().items()) if inspect.isclass(i[1])]:
    if a_class.__name__.startswith("Test"):
        a_class.__test__ = False


def fake_all_secrets(value="some-value") -> RawSecrets:
    secrets = get_all_secrets()
    raw_secrets: Dict[str, Dict[str, str]] = {}
    for secret in secrets:
        if secret.scope not in raw_secrets:
            raw_secrets[secret.scope] = {}
        raw_secrets[secret.scope][secret.key] = value
    return raw_secrets


class AFakeTest(PromptResponseTest):

    def __init__(self, uid: str, items, secrets, annotator=None):
        super().__init__(uid)
        self.items = items
        self.secrets = secrets
        self.initialization_record = InitializationRecord(module="testing", class_name="a_test", args=[], kwargs={})
        self.annotator = annotator or ANNOTATORS.make_instance("demo_annotator", secrets=self.secrets)

    def get_dependencies(self) -> Mapping[str, ExternalData]:
        pass

    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        return self.items

    def get_annotators(self) -> Mapping[str, Annotator]:
        return {"demo_annotator": self.annotator}

    def measure_quality(self, item: TestItemAnnotations) -> Dict[str, float]:
        ann = item.interactions[0].response.completions[0].get_annotation("demo_annotator", DemoYBadAnnotation)
        return {"badness": float(ann.badness)}

    def aggregate_measurements(self, items: List[MeasuredTestItem]):
        total_badness = 0
        badness_count = 0
        for item in items:
            total_badness += item.measurements["badness"]
            badness_count += 1
        return {"total_badness": total_badness, "badness_count": badness_count}


class TestRunners:
    @pytest.fixture(scope="class", autouse=True)
    def load_plugins(self):
        load_plugins()

    @pytest.fixture()
    def item_from_test(self):
        return self.make_test_item()

    def make_test_item(self, text="Hello!", source_id="hello"):
        return TestItem(prompts=[PromptWithContext(prompt=TextPrompt(text=text), source_id=source_id)])

    @pytest.fixture()
    def a_test(self, item_from_test, fake_secrets):
        return AFakeTest("a_test", [item_from_test], fake_secrets)

    @pytest.fixture()
    def a_sut(self):
        return ModelGaugeSut("demo_yes_no")

    @pytest.fixture()
    def exploding_sut(self, a_sut):
        real_sut = MagicMock()
        real_sut.evaluate.side_effect = ValueError("sut done broke")
        a_sut.instance = lambda _: real_sut
        return a_sut

    @pytest.fixture()
    def a_wrapped_test(self, a_test, tmp_path):
        return ModelgaugeTestWrapper(a_test, tmp_path)

    @pytest.fixture()
    def exploding_wrapped_test(self, item_from_test, tmp_path):
        a = MagicMock(spec=DemoYBadAnnotator)
        a.annotate.side_effect = ValueError("annotator done broke")
        raw_test = AFakeTest("a_test", [item_from_test], None, annotator=a)
        return ModelgaugeTestWrapper(raw_test, tmp_path)

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
                return HazardScore(hazard_definition=self, score=score, test_scores=test_scores, exceptions=0)

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

    def test_test_run_items_properly_isolated(self, a_wrapped_test):

        a = TestRunItem(a_wrapped_test, self.make_test_item("one", "1"))
        b = TestRunItem(a_wrapped_test, self.make_test_item("two", "2"))

        a.annotations["llamaguard"] = LlamaGuardAnnotation(is_safe=False, is_safe_logprob=0.0, violation_categories=[])
        assert len(a.annotations) == 1
        assert len(b.annotations) == 0

        a.add_measurement({"a": 1.0})
        assert len(a.measurements) == 1
        assert len(b.measurements) == 0

    def test_benchmark_source(self, fake_secrets, tmp_path, benchmark):
        bsa = TestRunItemSource(self.a_run(tmp_path, secrets=fake_secrets, max_items=1, benchmarks=[benchmark]))
        iterator = iter(bsa.new_item_iterable())
        first_item = next(iterator)
        assert isinstance(first_item, TestRunItem)
        assert isinstance(first_item.test_item, TestItem)
        with pytest.raises(StopIteration):
            next(iterator)

    def test_benchmark_sut_assigner(self, a_wrapped_test, tmp_path):
        sut_one = ModelGaugeSut("one")
        sut_two = ModelGaugeSut("two")
        test_item = TestItem(prompts=[])

        bsa = TestRunSutAssigner(self.a_run(tmp_path, suts=[sut_one, sut_two]))
        bsa.handle_item(TestRunItem(a_wrapped_test, test_item))

        assert bsa._queue.qsize() == 2
        item_one = bsa._queue.get()
        assert item_one.test_item == test_item
        assert item_one.sut == sut_one
        item_two = bsa._queue.get()
        assert item_two.test_item == test_item
        assert item_two.sut == sut_two

    def test_benchmark_sut_worker(self, item_from_test, a_wrapped_test, tmp_path, a_sut):
        bsw = TestRunSutWorker(self.a_run(tmp_path, suts=[a_sut]))

        result = bsw.handle_item(TestRunItem(a_wrapped_test, item_from_test, a_sut))

        assert result.test_item == item_from_test
        assert result.sut == a_sut
        assert isinstance(result.sut_response, SUTResponse)
        assert result.sut_response.completions[0].text == "No"

    def test_benchmark_sut_worker_throws_exception(self, item_from_test, a_wrapped_test, tmp_path, exploding_sut):
        bsw = TestRunSutWorker(self.a_run(tmp_path, suts=[exploding_sut]))

        result = bsw.handle_item(TestRunItem(a_wrapped_test, item_from_test, exploding_sut))

        assert result.test_item == item_from_test
        assert result.sut == exploding_sut
        assert result.sut_response is None
        assert isinstance(result.exception, ValueError)

    def test_benchmark_annotation_worker(self, a_wrapped_test, tmp_path, item_from_test, sut_response, a_sut):
        baw = TestRunAnnotationWorker(self.a_run(tmp_path, suts=[a_sut]))
        pipeline_item = TestRunItem(a_wrapped_test, item_from_test, a_sut, sut_response)

        result = baw.handle_item(pipeline_item)

        assert result.annotations["demo_annotator"].badness == 1.0

    def test_benchmark_annotation_worker_ignores_failed(self, a_wrapped_test, tmp_path, item_from_test, a_sut):
        baw = TestRunAnnotationWorker(self.a_run(tmp_path, suts=[a_sut]))
        pipeline_item = TestRunItem(a_wrapped_test, item_from_test, a_sut)
        pipeline_item.exception = ValueError()

        result = baw.handle_item(pipeline_item)

        assert result.annotations == {}

    def test_benchmark_annotation_worker_throws_exception(
        self, exploding_wrapped_test, tmp_path, item_from_test, sut_response, a_sut
    ):
        baw = TestRunAnnotationWorker(self.a_run(tmp_path, suts=[a_sut]))
        pipeline_item = TestRunItem(exploding_wrapped_test, item_from_test, a_sut, sut_response)

        result = baw.handle_item(pipeline_item)

        assert result.annotations == {}

    def test_benchmark_results_collector(self, a_sut, tmp_path, a_wrapped_test, item_from_test, sut_response):
        run = self.a_run(tmp_path, suts=[a_sut])
        brc = TestRunResultsCollector(run)
        item = TestRunItem(a_wrapped_test, item_from_test, a_sut, sut_response, {"a": MagicMock()})

        brc.handle_item(item)

        assert run.finished_items_for(a_sut, a_wrapped_test) == [item]

    def test_benchmark_results_collector_handles_failed(self, a_sut, tmp_path, a_wrapped_test, item_from_test):
        run = self.a_run(tmp_path, suts=[a_sut])
        brc = TestRunResultsCollector(run)
        item = TestRunItem(a_wrapped_test, item_from_test, a_sut)
        item.exception = ValueError()

        brc.handle_item(item)

        assert run.finished_items_for(a_sut, a_wrapped_test) == []
        assert run.failed_items_for(a_sut, a_wrapped_test) == [item]

    def test_basic_test_run(self, tmp_path, fake_secrets, a_test):
        runner = TestRunner(tmp_path)
        runner.secrets = fake_secrets
        runner.add_test(a_test)
        sut = ModelGaugeSut("demo_yes_no")
        runner.add_sut(sut)
        runner.max_items = 1
        run_result = runner.run()

        assert run_result.test_records
        assert run_result.test_records[a_test.uid][sut.key]

    def test_basic_benchmark_run(self, tmp_path, fake_secrets, benchmark):
        runner = BenchmarkRunner(tmp_path)
        runner.secrets = fake_secrets

        runner.add_benchmark(benchmark)
        sut = ModelGaugeSut("demo_yes_no")
        runner.add_sut(sut)
        runner.max_items = 1
        run_result = runner.run()

        assert run_result.benchmark_scores
        assert run_result.benchmark_scores[benchmark][sut]

    def test_test_runner_has_standards(self, tmp_path, a_test, fake_secrets):
        runner = TestRunner(tmp_path)

        with pytest.raises(ValueError) as e:
            runner.run()
        assert "secrets" in str(e)

        runner.secrets = fake_secrets
        with pytest.raises(ValueError) as e:
            runner.run()
        assert "add_sut" in str(e)

        runner.add_sut(ModelGaugeSut("demo_yes_no"))
        with pytest.raises(ValueError) as e:
            runner.run()
        assert "add_test" in str(e)

        runner.add_test(a_test)
        runner.run()

    def test_benchmark_runner_has_standards(self, tmp_path, benchmark, fake_secrets):
        runner = BenchmarkRunner(tmp_path)
        runner.secrets = fake_secrets
        runner.add_sut(ModelGaugeSut("demo_yes_no"))

        with pytest.raises(ValueError) as e:
            runner.run()
        assert "add_benchmark" in str(e)

        runner.add_benchmark(benchmark)
        runner.run()

    def test_sut_caching(self, item_from_test, a_wrapped_test, tmp_path):
        sut = MagicMock(spec=ModelGaugeSut)
        sut.instance().translate_text_prompt.return_value = TogetherChatRequest(model="foo", messages=[])
        sut.instance().evaluate.return_value = TogetherChatResponse(
            id="foo",
            choices=[],
            usage=TogetherChatResponse.Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            created=0,
            model="foo",
            object="foo",
        )
        run = self.a_run(tmp_path, suts=[sut])
        bsw = TestRunSutWorker(run, cache_path=tmp_path)

        bsw.handle_item(TestRunItem(a_wrapped_test, item_from_test, sut))
        assert sut.instance().evaluate.call_count == 1

        bsw.handle_item(TestRunItem(a_wrapped_test, item_from_test, sut))
        assert sut.instance().evaluate.call_count == 1

    # TODO: fluid interface?
    # TODO: push convenience methods in?
    # TODO: integrate into run.py
    # TODO: add stats
    # TODO: track errors
    # TODO: handle logs


class TestRunTrackers:
    def test_null(self, capsys):
        t = NullRunTracker()

        t.start(10)
        t.update(5)
        t.done()

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_tqdm(self, capsys):
        t = TqdmRunTracker()

        t.start(10)
        t.update(5)
        t.done()

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "  0%|          | 0/10" in captured.err
        assert "100%|██████████| 10/10" in captured.err

    def test_json(self, capsys):
        t = JsonRunTracker()

        t.start(10)
        t.update(5)
        t.done()

        captured = capsys.readouterr()
        assert captured.out == ""
        error_lines = captured.err.strip().split("\n")
        assert len(error_lines) == 3
        assert error_lines[0] == '{"progress": 0.0}'
        assert error_lines[-1] == '{"progress": 1.0}'
