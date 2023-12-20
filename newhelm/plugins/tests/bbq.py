from typing import List
from newhelm.annotation import AnnotatedInteraction
from newhelm.base_test import BasePromptResponseTest
from newhelm.placeholders import Prompt, Result


class BBQ(BasePromptResponseTest):
    def make_prompts(self) -> List[Prompt]:
        # In the real thing, this would use an ExampleImporter and Adapters
        lines = [
            "The following are multiple choice questions (with answers).",
            "Passage: Is this the BBQ Test? Answer: Yes",
            "Passage: is this the real BBQ? Answer:",
        ]
        return [Prompt("\n".join(lines))]

    def calculate_results(
        self, interactions: List[AnnotatedInteraction]
    ) -> List[Result]:
        # In the real thing, this would be handled by Metric objects
        prompt_longer_count = 0
        for interaction in interactions:
            first_turn = interaction.interaction.turns[0]
            if len(first_turn.completion) < len(first_turn.prompt.text):
                prompt_longer_count += 1
        return [
            Result(
                name="prompt_longer_count",
                value=prompt_longer_count / len(interactions),
            )
        ]
