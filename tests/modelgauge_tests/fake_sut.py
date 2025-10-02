from modelgauge.prompt import ChatPrompt, TextPrompt
from modelgauge.sut import PromptResponseSUT, SUTOptions, SUTResponse
from modelgauge.sut_capabilities import AcceptsChatPrompt, AcceptsTextPrompt
from modelgauge.sut_decorator import modelgauge_sut
from pydantic import BaseModel


class FakeSUTRequest(BaseModel):
    text: str


class FakeSUTResponse(BaseModel):
    text: str


@modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
class FakeSUT(PromptResponseSUT):
    """SUT that just echos the prompt text back."""

    def __init__(self, uid: str = "fake-sut"):
        super().__init__(uid)
        self.evaluate_calls = 0

    def translate_text_prompt(self, prompt: TextPrompt, options: SUTOptions) -> FakeSUTRequest:
        return FakeSUTRequest(text=prompt.text)

    def translate_chat_prompt(self, prompt: ChatPrompt, options: SUTOptions) -> FakeSUTRequest:
        return FakeSUTRequest(text=prompt.messages[-1].text)

    def evaluate(self, request: FakeSUTRequest) -> FakeSUTResponse:
        self.evaluate_calls += 1
        return FakeSUTResponse(text=request.text)

    def translate_response(self, request: FakeSUTRequest, response: FakeSUTResponse) -> SUTResponse:
        return SUTResponse(text=response.text)


@modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
class BadSUT(FakeSUT):
    """SUT whose evaluate always raises an exception."""

    def evaluate(self, request: FakeSUTRequest) -> FakeSUTResponse:
        raise RuntimeError("SUT failed to evaluate")
