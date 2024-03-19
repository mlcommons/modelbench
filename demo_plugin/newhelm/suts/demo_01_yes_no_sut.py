from pydantic import BaseModel
from newhelm.prompt import ChatPrompt, TextPrompt
from newhelm.prompt_formatting import format_chat
from newhelm.sut import SUTCompletion, SUTResponse, PromptResponseSUT
from newhelm.sut_registry import SUTS


class DemoYesNoRequest(BaseModel):
    """The behavior of this sut only depends on the Prompt text."""

    text: str


class DemoYesNoResponse(BaseModel):
    """This SUT is only capable of returning text."""

    number_of_words: int
    text: str


class DemoYesNoSUT(PromptResponseSUT[DemoYesNoRequest, DemoYesNoResponse]):
    """This SUT demonstrates the bare minimum behavior of a SUT: Use the input Prompt to determine the response."""

    def translate_text_prompt(self, prompt: TextPrompt) -> DemoYesNoRequest:
        return DemoYesNoRequest(text=prompt.text)

    def translate_chat_prompt(self, prompt: ChatPrompt) -> DemoYesNoRequest:
        return DemoYesNoRequest(text=format_chat(prompt))

    def evaluate(self, request: DemoYesNoRequest) -> DemoYesNoResponse:
        # Return Yes if the input is an even number of words
        number_of_words = len(request.text.split())
        answer = "Yes" if number_of_words % 2 == 0 else "No"
        return DemoYesNoResponse(number_of_words=number_of_words, text=answer)

    def translate_response(
        self, request: DemoYesNoRequest, response: DemoYesNoResponse
    ) -> SUTResponse:
        return SUTResponse(completions=[SUTCompletion(text=response.text)])


SUTS.register(DemoYesNoSUT, "demo_yes_no")
