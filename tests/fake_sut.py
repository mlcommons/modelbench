from typing import List
from pydantic import BaseModel
from newhelm.prompt import ChatPrompt, TextPrompt
from newhelm.sut import PromptResponseSUT, SUTCompletion, SUTResponse
from newhelm.sut_capabilities import AcceptsChatPrompt, AcceptsTextPrompt
from newhelm.sut_decorator import newhelm_sut


class FakeSUTRequest(BaseModel):
    text: str
    num_completions: int


class FakeSUTResponse(BaseModel):
    completions: List[str]


@newhelm_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
class FakeSUT(PromptResponseSUT[FakeSUTRequest, FakeSUTResponse]):
    """SUT that just echos the prompt text back."""

    def __init__(self, uid: str = "fake-sut"):
        super().__init__(uid)
        self.evaluate_calls = 0

    def translate_text_prompt(self, prompt: TextPrompt) -> FakeSUTRequest:
        return FakeSUTRequest(
            text=prompt.text, num_completions=prompt.options.num_completions
        )

    def translate_chat_prompt(self, prompt: ChatPrompt) -> FakeSUTRequest:
        return FakeSUTRequest(
            text=prompt.messages[-1].text,
            num_completions=prompt.options.num_completions,
        )

    def evaluate(self, request: FakeSUTRequest) -> FakeSUTResponse:
        self.evaluate_calls += 1
        completions = []
        for _ in range(request.num_completions):
            completions.append(request.text)
        return FakeSUTResponse(completions=completions)

    def translate_response(
        self, request: FakeSUTRequest, response: FakeSUTResponse
    ) -> SUTResponse:
        completions = []
        for text in response.completions:
            completions.append(SUTCompletion(text=text))
        return SUTResponse(completions=completions)
