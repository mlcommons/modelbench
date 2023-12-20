from typing import List
from newhelm.placeholders import Prompt
from newhelm.sut import SUTResponse, PromptResponseSUT


class GPT2(PromptResponseSUT):
    """The SUT should have all the details currently spread across model_deployment and model_metadata."""

    def evaluate(self, prompt: Prompt) -> SUTResponse:
        # Pure placeholder.
        number_of_words = len(prompt.text.split())
        return SUTResponse(f"The prompt has {number_of_words} words.")
