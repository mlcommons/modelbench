import dataclasses
import pathlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, List

from modelgauge.annotation import Annotation
from modelgauge.annotator import CompletionAnnotator
from modelgauge.base_test import PromptResponseTest
from modelgauge.dependency_helper import FromSourceDependencyHelper
from modelgauge.external_data import WebData
from modelgauge.single_turn_prompt_response import (
    TestItem,
    PromptWithContext,
    MeasuredTestItem,
    TestItemAnnotations,
    PromptInteractionAnnotations,
    SUTResponseAnnotations,
    SUTCompletionAnnotations,
)
from modelgauge.sut import PromptResponseSUT, SUTResponse, SUTCompletion


# in their own file to solve circular import problems


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

    def make_test_items(self) -> List[TestItem]:
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

    def sut_options(self):
        """This is ridiculous but necessary for the moment."""
        return self.make_test_items()[0].prompts[0].prompt.options

    def dependencies(self):
        result = {}
        if self.dependency_helper.dependencies:
            for k, v in self.dependency_helper.dependencies.items():
                if isinstance(v, WebData):
                    result[k] = {"source": v.source_url}
                    result[k]["local_path"] = self.dependency_helper.get_local_path(k)
                    path = pathlib.Path(self.dependency_helper.get_local_path(k))
                    if path.exists():
                        result[k]["timestamp"] = datetime.fromtimestamp(
                            path.stat().st_mtime, tz=timezone.utc
                        ).isoformat()
                else:
                    result[k] = str(v)
        return result

    def __repr__(self):
        return f"{self.__class__.__name__}(uid={self.uid})"


@dataclass
class TestRunItem:
    """The data related to running a single test item"""

    test: ModelgaugeTestWrapper
    test_item: TestItem
    sut: PromptResponseSUT = None
    sut_response: SUTResponse = None
    annotations: dict[str, Annotation] = dataclasses.field(default_factory=dict)
    measurements: dict[str, float] = dataclasses.field(default_factory=dict)
    exceptions: list = dataclasses.field(default_factory=list)

    def prompt_with_context(self) -> PromptWithContext:
        return self.test_item.prompts[0]

    def completion(self) -> SUTCompletion:
        if self.sut_response and self.sut_response.completions:
            return self.sut_response.completions[0]

    def add_measurement(self, measurement: dict):
        self.measurements.update(measurement)

    def source_id(self):
        return self.prompt_with_context().source_id


class Timer:

    def __init__(self):
        super().__init__()
        self.elapsed = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.start
