from typing import List
from newhelm.base_test import BasePromptResponseTest
from newhelm.placeholders import Measurement, Prompt, Result
from newhelm.single_turn_prompt_response import (
    AnnotatedTestItem,
    MeasuredTestItem,
    PromptWithContext,
    TestItem,
)


class MMLU(BasePromptResponseTest):
    def make_test_items(self) -> List[TestItem]:
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
