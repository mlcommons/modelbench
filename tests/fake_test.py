from newhelm.annotator import Annotator
from newhelm.base_test import PromptResponseTest
from newhelm.dependency_helper import DependencyHelper
from newhelm.external_data import ExternalData
from newhelm.prompt import TextPrompt
from newhelm.single_turn_prompt_response import (
    MeasuredTestItem,
    PromptWithContext,
    TestItem,
    TestItemAnnotations,
)
from newhelm.sut_capabilities import AcceptsTextPrompt
from newhelm.test_decorator import newhelm_test
from pydantic import BaseModel
from typing import Dict, List, Mapping


def fake_test_item(text):
    """Create a TestItem with `text` as the prompt text."""
    return TestItem(
        prompts=[PromptWithContext(prompt=TextPrompt(text=text), source_id=None)]
    )


class FakeTestResult(BaseModel):
    count_test_items: int


@newhelm_test(requires_sut_capabilities=[AcceptsTextPrompt])
class FakeTest(PromptResponseTest):
    """Test that lets the user override almost all of the behavior."""

    def __init__(
        self,
        uid: str = "test-uid",
        *,
        dependencies={},
        test_items=[],
        annotators={},
        measurement={}
    ):
        super().__init__(uid)
        self.dependencies = dependencies
        self.test_items = test_items
        self.annotators = annotators
        self.measurement = measurement

    def get_dependencies(self) -> Mapping[str, ExternalData]:
        return self.dependencies

    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        return self.test_items

    def get_annotators(self) -> Mapping[str, Annotator]:
        return self.annotators

    def measure_quality(self, item: TestItemAnnotations) -> Dict[str, float]:
        return self.measurement

    def aggregate_measurements(self, items: List[MeasuredTestItem]) -> FakeTestResult:
        return FakeTestResult(count_test_items=len(items))
