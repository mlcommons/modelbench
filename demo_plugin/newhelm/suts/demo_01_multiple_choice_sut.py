from typing import List, TypedDict
from newhelm.placeholders import Prompt
from newhelm.sut import SUTCompletion, SUTResponse, PromptResponseSUT
from newhelm.sut_registry import SUTS


class DemoMultipleChoiceSUTRequest(TypedDict):
    """The behavior of this sut only depends on the Prompt text."""

    text: str


class DemoMultipleChoiceSUTResponse(TypedDict):
    """This SUT is only capable of returning text."""

    text: str


class DemoMultipleChoiceSUT(
    PromptResponseSUT[DemoMultipleChoiceSUTRequest, DemoMultipleChoiceSUTResponse]
):
    """This SUT demonstrates the bare minimum behavior of a SUT: Use the input Prompt to determine the response."""

    def translate_request(self, prompt: Prompt) -> DemoMultipleChoiceSUTRequest:
        return {"text": prompt.text}

    def evaluate(
        self, request: DemoMultipleChoiceSUTRequest
    ) -> DemoMultipleChoiceSUTResponse:
        # Pick a letter A, B, C, or D based on prompt length.
        number_of_words = len(request["text"].split())
        return {"text": chr(ord("A") + number_of_words % 4)}

    def translate_response(
        self, prompt: Prompt, response: DemoMultipleChoiceSUTResponse
    ) -> SUTResponse:
        return SUTResponse([SUTCompletion(response["text"])])


SUTS.register("DemoMultipleChoiceSUT", DemoMultipleChoiceSUT)
