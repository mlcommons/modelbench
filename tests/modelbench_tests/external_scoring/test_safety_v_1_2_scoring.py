from typing import Sequence, Mapping, List

from modelbench.benchmark_runner_items import TestRunItem
from modelbench.benchmarks import BenchmarkDefinition
from modelbench.external_scoring.scoring_adapters import RegressionScorer
from modelbench.hazards import HazardDefinition
from modelbench_tests.test_run import fake_benchmark_run
from modelgauge.hazards import Hazards
from modelgauge.prompt import TextPrompt
from modelgauge.records import TestRecord
from modelgauge.single_turn_prompt_response import TestItem
from modelgauge.tests.safe_v1 import SafeTestItemContext
from modelgauge_tests.fake_sut import FakeSUT


class FakeHazard(HazardDefinition):

    def __init__(self, hazard_key):
        super().__init__()
        self.hazard_key = hazard_key
        # self.locale = "en_us"

    def test_uids(self) -> List[str]:
        pass

    def score(self, sut_scores: Mapping[str, TestRecord]) -> "HazardScore":
        pass


def test_basic_scoring(tmp_path):
    scorer = RegressionScorer()
    hazards = [FakeHazard(key) for key in Hazards()]

    class ABenchmark(BenchmarkDefinition):
        VERSION = "1.0"

        def _make_hazards(self) -> Sequence[HazardDefinition]:
            return hazards

        _uid_definition = {"class": "self", "version": "0.5"}

        @property
        def reference_suts(self) -> list[str]:
            return ["demo_yes_no"]

    benchmark = ABenchmark()
    sut = FakeSUT("some_sut")
    run = fake_benchmark_run(benchmark, hazards, sut, tmp_path)
    example_items: list[TestRunItem] = []
    for k1 in run.finished_items:
        for k2 in run.finished_items[k1]:
            example_items.extend(run.finished_items[k1][k2])
    for item in example_items:
        item.test_item.context_internal = SafeTestItemContext(
            persona_type="whatever", hazard="cse", difficulty_b=0.1, discrimination_a=0.2
        )
    example = example_items[0]
    good_hazards = Hazards()._known_hazards - {"spc", "sxc"}
    finished_items: list[TestRunItem] = [
        TestRunItem(
            example.test,
            TestItem(
                prompt=TextPrompt(text="ignored"),
                source_id=f"prompt-{index}",
                context=SafeTestItemContext(
                    persona_type="whatever",
                    hazard=hazard,
                    difficulty_b=0.1,
                    discrimination_a=0.2,
                ),
            ),
            sut,
            example.sut_response,
            example.annotations,
            example.measurements,
            example.exceptions,
        )
        for index, hazard in enumerate(good_hazards)
    ]

    result = scorer.score(benchmark, sut, finished_items)
    assert result.benchmark_definition == benchmark
    assert result.sut == sut
    assert {h.hazard_definition.hazard_key for h in result.hazard_scores} == set(Hazards())
    for h in result.hazard_scores:
        assert 0 < h.score < 100
    assert 0 < result.score < 100
