from typing import List
from newhelm.placeholders import Prompt
from newhelm.sut import SUTResponse, PromptResponseSUT
from newhelm.sut_registry import SUTS


class DemoMultipleChoiceSUT(PromptResponseSUT):
    """This SUT demonstrates the bare minimum behavior of a SUT: Use the input Prompt to determine the response."""

    def evaluate(self, prompt: Prompt) -> SUTResponse:
        # Pick a letter A, B, C, or D based on prompt length.
        number_of_words = len(prompt.text.split())
        return SUTResponse(chr(ord("A") + number_of_words % 4))


SUTS.register("DemoMultipleChoiceSUT", DemoMultipleChoiceSUT())
