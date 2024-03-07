from typing import Dict, List, Mapping
from newhelm.base_annotator import BaseAnnotator
from newhelm.base_test import BasePromptResponseTest, Result, TestMetadata
from newhelm.dependency_helper import DependencyHelper
from newhelm.external_data import ExternalData
from newhelm.prompt import TextPrompt
from newhelm.record_init import record_init
from newhelm.single_turn_prompt_response import (
    MeasuredTestItem,
    PromptWithContext,
    TestItem,
    TestItemAnnotations,
)


def fake_test_item(text):
    """Create a TestItem with `text` as the prompt text."""
    return TestItem(
        prompts=[PromptWithContext(prompt=TextPrompt(text=text), source_id=None)]
    )


class FakeTest(BasePromptResponseTest):
    """Test that lets the user override almost all of the behavior."""

    @record_init
    def __init__(
        self, *, dependencies={}, test_items=[], annotators={}, measurement={}
    ):
        self.dependencies = dependencies
        self.test_items = test_items
        self.annotators = annotators
        self.measurement = measurement

    def get_metadata(self) -> TestMetadata:
        """Return a description of the test."""
        return TestMetadata(name="fake-test", description="fake-description")

    def get_dependencies(self) -> Mapping[str, ExternalData]:
        return self.dependencies

    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        return self.test_items

    def get_annotators(self) -> Mapping[str, BaseAnnotator]:
        return self.annotators

    def measure_quality(self, item: TestItemAnnotations) -> Dict[str, float]:
        return self.measurement

    def aggregate_measurements(self, items: List[MeasuredTestItem]) -> List[Result]:
        return [Result(name="count_test_items", value=len(items))]
