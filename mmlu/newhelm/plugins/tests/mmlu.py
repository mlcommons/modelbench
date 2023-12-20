from typing import List
from newhelm.annotation import AnnotatedInteraction
from newhelm.base_test import BasePromptResponseTest, BaseTest
from newhelm.placeholders import Prompt, Result


class MMLU(BasePromptResponseTest):
    def make_prompts(self) -> List[Prompt]:
        # In the real thing, this would use an ExampleImporter and Adapters
        return [
            Prompt("When I think of MMLU, the word that comes to mind is"),
            Prompt("But the worst part is when"),
        ]

    def calculate_results(
        self, interactions: List[AnnotatedInteraction]
    ) -> List[Result]:
        # In the real thing, this would be handled by Metric objects
        return [Result("count", value=len(interactions))]
