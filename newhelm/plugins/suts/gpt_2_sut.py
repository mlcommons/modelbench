from typing import List
from newhelm.placeholders import Prompt
from newhelm.sut import Interaction, PromptResponseSUT, Turn


class GPT2(PromptResponseSUT):
    """The SUT should have all the details currently spread across model_deployment and model_metadata."""

    def evaluate(self, prompt: Prompt) -> Interaction:
        # Pure placeholder.
        number_of_words = len(prompt.text.split())
        return Interaction([Turn(prompt, f"The prompt has {number_of_words} words.")])
