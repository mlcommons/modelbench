import pytest

from pydantic import BaseModel

from modelgauge.model_options import ModelOptions
from modelgauge.prompt import TextPrompt
from modelgauge.reasoning_handlers import ReasoningRequest, ThinkingMixin

from modelgauge.sut import SUTResponse, PromptResponseSUT
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.sut_decorator import modelgauge_sut


class FakeSUTRequest(BaseModel):
    text: str
    max_tokens: int


class FakeSUTResponse(BaseModel):
    text: str


class FakeBaseSUT(PromptResponseSUT):
    def __init__(self, uid: str = "fake-sut"):
        super().__init__(uid)

    def translate_text_prompt(self, prompt: TextPrompt, options: ModelOptions) -> FakeSUTRequest:
        return FakeSUTRequest(text="prompt", max_tokens=options.max_tokens)

    def evaluate(self, request: FakeSUTRequest) -> FakeSUTResponse:
        return FakeSUTResponse(text="reasoning</think>response")

    def translate_response(self, request: FakeSUTRequest, response: FakeSUTResponse) -> SUTResponse:
        return SUTResponse(text=response.text)


class TestThinkMixin:
    @pytest.fixture
    def sut(self):
        @modelgauge_sut(capabilities=[AcceptsTextPrompt])
        class ThinkSut(ThinkingMixin, FakeBaseSUT):
            pass

        return ThinkSut("sut-uid")

    def test_translate_text_prompt_sets_max_tokens(self, sut):
        prompt = TextPrompt(text="some-text")

        options = ModelOptions(max_tokens=50)
        request = sut.translate_text_prompt(prompt, options)
        assert request.request.max_tokens == 50
        assert request.max_content_tokens == 50

        options = ModelOptions(max_tokens=50, max_total_output_tokens=200)
        request = sut.translate_text_prompt(prompt, options)
        assert request.request.max_tokens == 200
        assert request.max_content_tokens == 50

        options = ModelOptions(max_total_output_tokens=200)
        request = sut.translate_text_prompt(prompt, options)
        assert request.request.max_tokens == 200
        assert request.max_content_tokens == None  # Default max tokens

    @pytest.mark.parametrize(
        "full_text, content_text",
        [("<think>hmm</think>\n Output", "Output"), ("hmm</think>\n Output", "Output"), ("<think>hmmm", "")],
    )
    def test_translate_response_no_truncation(self, full_text, content_text, sut):
        request = ReasoningRequest(request=FakeSUTRequest(text="", max_tokens=100), max_content_tokens=100)
        response = FakeSUTResponse(text=full_text)

        result = sut.translate_response(request, response)
        assert result.text == content_text

    @pytest.mark.parametrize(
        "full_text, content_text",
        [
            ("<think>hmm</think>one two three", "one two"),
            ("<think></think>one", "one"),
            ("<think></think>", ""),
            ("<think>hmmm", ""),
        ],
    )
    def test_truncation(self, full_text, content_text, sut):
        request = ReasoningRequest(request=FakeSUTRequest(text="", max_tokens=100), max_content_tokens=2)
        response = FakeSUTResponse(text=full_text)

        result = sut.translate_response(request, response)
        assert result.text == content_text

    def test_translate_response_warns_reasoning_over_budget(self, sut, caplog):
        request = ReasoningRequest(request=FakeSUTRequest(text="", max_tokens=5), max_content_tokens=2)
        response = FakeSUTResponse(text="one two three</think>four five")

        result = sut.translate_response(request, response)
        assert "reasoning likely ate into the token budget of the actual output" in caplog.text
