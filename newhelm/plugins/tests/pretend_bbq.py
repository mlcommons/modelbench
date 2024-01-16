from collections import defaultdict
from typing import Dict, List, Mapping
from newhelm.base_test import BasePromptResponseTest, TestMetadata
from newhelm.dependency_helper import DependencyHelper
from newhelm.external_data import ExternalData
from newhelm.placeholders import Prompt, Result
from newhelm.single_turn_prompt_response import (
    AnnotatedTestItem,
    MeasuredTestItem,
    PromptWithContext,
    TestItem,
)


class PretendBBQ(BasePromptResponseTest):
    @classmethod
    def get_metadata(cls) -> TestMetadata:
        """Return a description of the test."""
        return TestMetadata(
            "MMLU", "This is a fake BBQ to demonstrate how some stuff works."
        )

    @classmethod
    def get_dependencies(cls) -> Mapping[str, ExternalData]:
        """No external dependencies."""
        return {}

    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        # In the real thing, this would use DependencyHelper and other libraries.
        lines = [
            "The following are multiple choice questions (with answers).",
            "Passage: Is this the BBQ Test? Answer: Yes",
            "Passage: is this the real BBQ? Answer:",
        ]
        return [
            TestItem(
                [
                    PromptWithContext(
                        prompt=Prompt("\n".join(lines)),
                        context="No",
                    )
                ]
            )
        ]

    def measure_quality(self, item: AnnotatedTestItem) -> Dict[str, float]:
        interaction = item.item_with_interactions.interactions[0]
        # Check if the right answer is in the response
        correct = 0
        if interaction.prompt.context in interaction.response.completion:
            correct = 1
        return {"correct": correct}

    def aggregate_measurements(self, items: List[MeasuredTestItem]) -> List[Result]:
        # In the real thing, this would be handled by library functions
        grouped_by_name = defaultdict(list)
        for item in items:
            for name, measurement in item.measurements.items():
                grouped_by_name[name].append(measurement)

        results = []
        for key, values in grouped_by_name.items():
            results.append(Result(key, sum(values) / len(values)))
        return results
