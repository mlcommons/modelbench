from typing import List
from newhelm.placeholders import Prompt
from newhelm.sut import SUTResponse, PromptResponseSUT
from newhelm.sut_registry import SUTS


class GPT2(PromptResponseSUT):
    """The SUT should have all the details currently spread across model_deployment and model_metadata."""

    def evaluate(self, prompt: Prompt) -> SUTResponse:
        # Pure placeholder.
        number_of_words = len(prompt.text.split())
        # Pick a letter A, B, C, or D based on prompt length.
        return SUTResponse(chr(ord("A") + number_of_words % 4))


SUTS.register("gpt2", GPT2())
