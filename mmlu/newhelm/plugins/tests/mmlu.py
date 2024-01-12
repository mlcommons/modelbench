from typing import List, Mapping
from newhelm.base_test import BasePromptResponseTest, TestMetadata
from newhelm.dependency_helper import DependencyHelper
from newhelm.external_data import ExternalData
from newhelm.placeholders import Measurement, Prompt, Result
from newhelm.single_turn_prompt_response import (
    AnnotatedTestItem,
    MeasuredTestItem,
    PromptWithContext,
    TestItem,
)
from newhelm.test_registry import TESTS


class MMLU(BasePromptResponseTest):
    @classmethod
    def get_metadata(cls) -> TestMetadata:
        """Return a description of the test."""
        return TestMetadata(
            "MMLU", "This is a fake MMLU to demonstrate how some stuff works."
        )

    @classmethod
    def get_dependencies(cls) -> Mapping[str, ExternalData]:
        """No external dependencies."""
        return {}

    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        # In the real thing, this would use DependencyHelper and other libraries.
        return [
            TestItem(
                [
                    PromptWithContext(
                        Prompt("When I think of MMLU, the word that comes to mind is")
                    )
                ]
            ),
            TestItem([PromptWithContext(Prompt("But the worst part is when"))]),
        ]

    def measure_quality(self, item: AnnotatedTestItem) -> List[Measurement]:
        return [Measurement("count", 1)]

    def aggregate_measurements(self, items: List[MeasuredTestItem]) -> List[Result]:
        # In the real thing, this would be handled by Metric objects
        total = 0.0
        for item in items:
            for measurement in item.measurements:
                total += measurement.value
        return [Result("total", value=total)]


TESTS.register("mmlu", MMLU())
