from collections import defaultdict
from typing import List
from newhelm.base_test import BasePromptResponseTest
from newhelm.placeholders import Measurement, Prompt, Result
from newhelm.single_turn_prompt_response import (
    AnnotatedTestItem,
    MeasuredTestItem,
    PromptWithContext,
    TestItem,
)


class BBQ(BasePromptResponseTest):
    def make_test_items(self) -> List[TestItem]:
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

    def measure_quality(self, item: AnnotatedTestItem) -> List[Measurement]:
        interaction = item.item_with_interactions.interactions[0]
        # Check if the right answer is in the response
        correct = 0
        if interaction.prompt.context in interaction.response.completion:
            correct = 1
        return [Measurement("correct", correct)]

    def aggregate_measurements(self, items: List[MeasuredTestItem]) -> List[Result]:
        # In the real thing, this would be handled by library functions
        grouped_by_name = defaultdict(list)
        for item in items:
            for measurement in item.measurements:
                grouped_by_name[measurement.name].append(measurement.value)

        results = []
        for key, values in grouped_by_name.items():
            results.append(Result(key, sum(values) / len(values)))
        return results
