from typing import Dict, List, Mapping

from modelbench.benchmarks import BenchmarkDefinition
from modelbench.hazards import HazardDefinition, HazardScore
from modelbench.scoring import ValueEstimate

from modelgauge.base_test import PromptResponseTest
from modelgauge.annotators.demo_annotator import DemoYBadAnnotation
from modelgauge.dependency_helper import DependencyHelper
from modelgauge.external_data import ExternalData
from modelgauge.prompt import TextPrompt
from modelgauge.record_init import InitializationRecord
from modelgauge.records import TestRecord
from modelgauge.single_turn_prompt_response import MeasuredTestItem, SUTResponseAnnotations, TestItem

import modelgauge.annotators.testing_annotator  # required for the annotator registry to pick it up


REFERENCE_SUTS = ["sut_1", "sut_2"]


class AFakeTest(PromptResponseTest):
    def __init__(self, uid: str, items, annotators=["demo_annotator"]):
        super().__init__(uid)
        self.items = items
        self.initialization_record = InitializationRecord(module="testing", class_name="a_test", args=[], kwargs={})
        self.annotator_uids = annotators

    def get_dependencies(self) -> Mapping[str, ExternalData]:
        pass

    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        return self.items

    def get_annotators(self) -> List[str]:
        return self.annotator_uids

    def measure_quality(self, item: SUTResponseAnnotations) -> Dict[str, float]:
        if item.annotations:
            return {"badness": float(item.get_annotation(self.annotator_uids[0], DemoYBadAnnotation).badness)}
        else:
            return {}

    def aggregate_measurements(self, items: List[MeasuredTestItem]):
        total_badness = 0
        badness_count = 0
        for item in items:
            total_badness += item.measurements["badness"]
            badness_count += 1
        return {"total_badness": total_badness, "badness_count": badness_count}


class AHazard(HazardDefinition):
    def __init__(self, test_list):
        super().__init__()
        self._tests = test_list

    def test_uids(self) -> List[str]:
        return [test.uid for test in self._tests]

    def score(self, sut_scores: Mapping[str, TestRecord]) -> HazardScore:
        count = 0
        total = 0
        for key, value in sut_scores.items():
            d = value.result.data
            count += d["badness_count"]
            total += d["total_badness"]

        score = ValueEstimate.make(total / count, count)

        test_scores = {}
        return HazardScore(
            hazard_definition=self,
            score=score,
            test_scores=test_scores,
            exceptions=0,
            num_scored_items=10000,
            num_safe_items=9999,
        )


class DummyTest(AFakeTest):
    def __init__(self, uid: str, items, annotators=["testing_annotator"]):
        super().__init__(uid, items, annotators)

    def measure_quality(self, item):
        return {"is_safe": 0.0}

    def aggregate_measurements(self, items):
        return {"is_safe": 0.0}


class DummyHazard(HazardDefinition):
    def __init__(self, key):
        super().__init__()
        self._key = key
        self._tests = [DummyTest("test1", [TestItem(prompt=TextPrompt(text="text"), source_id="id")])]

    @property
    def reference_key(self):
        return self._key

    def test_uids(self) -> list[str]:
        return ["test1"]

    def score(self, sut_scores):
        return HazardScore(
            hazard_definition=self,
            score=ValueEstimate.make(0.0, 100),
            test_scores={},
            exceptions=0,
            num_scored_items=100,
            num_safe_items=0,
        )


class DummyHazard2(DummyHazard):
    """Different hazard with different UID"""

    pass


class DummyBenchmark(BenchmarkDefinition):

    def __init__(self, hazards, uid):
        self._hazards = hazards
        self._uid = uid
        super().__init__()

    @property
    def reference_suts(self):
        return ["sut_1", "sut_2"]

    def reference_benchmark(self):
        return DummyBenchmark(self._hazards, "reference_benchmark")

    def _make_hazards(self) -> list[HazardDefinition]:
        return self._hazards

    _uid_definition = {
        "uid": "self._uid",
    }
