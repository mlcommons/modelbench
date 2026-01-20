from modelgauge.prompt import ChatPrompt, TextPrompt
from modelgauge.sut import PromptResponseSUT, SUTResponse
from modelgauge.model_options import ModelOptions, TokenProbability, TopTokens
from modelgauge.sut_capabilities import AcceptsChatPrompt, AcceptsTextPrompt, ProducesPerTokenLogProbabilities
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

    def translate_text_prompt(self, prompt: TextPrompt, options: ModelOptions) -> FakeSUTRequest:
        return FakeSUTRequest(text=prompt.text)

    def translate_chat_prompt(self, prompt: ChatPrompt, options: ModelOptions) -> FakeSUTRequest:
        return FakeSUTRequest(text=prompt.messages[-1].text)

    def evaluate(self, request: FakeSUTRequest) -> FakeSUTResponse:
        self.evaluate_calls += 1
        return FakeSUTResponse(text=request.text)

    def translate_response(self, request: FakeSUTRequest, response: FakeSUTResponse) -> SUTResponse:
        return SUTResponse(text=response.text)


@modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt, ProducesPerTokenLogProbabilities])
class FakeSUTWithLogprobs(FakeSUT):
    def __init__(self, uid: str, logprobs: list[list[tuple[str, float]]]):
        super().__init__(uid)
        self.logprobs = logprobs

    def translate_response(self, request: FakeSUTRequest, response: FakeSUTResponse) -> SUTResponse:
        logprobs = []
        for token_logprobs in self.logprobs:
            logprobs.append(
                TopTokens(
                    top_tokens=[TokenProbability(token=token, logprob=logprob) for token, logprob in token_logprobs]
                )
            )
        # logprobs = [TopTokens(top_tokens=[TokenProbability(token=token, logprob=logprob)]) for token, logprob in self.logprobs]
        return SUTResponse(text=response.text, top_logprobs=logprobs)


@modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
class BadSUT(FakeSUT):
    """SUT whose evaluate always raises an exception."""

    def evaluate(self, request: FakeSUTRequest) -> FakeSUTResponse:
        raise RuntimeError("SUT failed to evaluate")
