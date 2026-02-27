import pytest
from unittest.mock import patch

from pydantic import BaseModel

from modelgauge.model_options import ModelOptions
from modelgauge.prompt import TextPrompt
from modelgauge.reasoning_handlers import ReasoningRequest, ReasoningSUT, ThinkingMixin

from modelgauge.sut import SUTResponse, PromptResponseSUT
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.sut_decorator import modelgauge_sut


class FakeSUTRequest(BaseModel):
    text: str
    max_tokens: int | None = None


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


class TestReasoningSUT:

    class CountMixin(ReasoningSUT, FakeBaseSUT):
        # Inherit from FakeBaseSUT so that this is a concrete class.
        @classmethod
        def response_contains_reasoning(cls, response: SUTResponse) -> bool:
            return "123" in response.text

    @pytest.fixture(autouse=True)
    def _patch_reasoning_suts(self):
        # Only consider the CountMixin for matching.
        with patch.object(
            ReasoningSUT,
            "_get_concrete_reasoning_suts",
            return_value={self.CountMixin},
        ):
            yield

    def test_find_thinking_mixin(self):
        class CountSUT(FakeBaseSUT):
            def evaluate(self, request: FakeSUTRequest) -> FakeSUTResponse:
                return FakeSUTResponse(text="123")

        sut = CountSUT("sut")
        reasoning_cls = ReasoningSUT.find_match(sut)
        assert reasoning_cls == self.CountMixin

    def test_find_no_match(self):
        class NoReasoningSUT(FakeBaseSUT):
            def evaluate(self, request: FakeSUTRequest) -> FakeSUTResponse:
                return FakeSUTResponse(text="text only")

        sut = NoReasoningSUT("sut")
        reasoning_cls = ReasoningSUT.find_match(sut)
        assert reasoning_cls is None


class TestThinkMixin:
    @modelgauge_sut(capabilities=[AcceptsTextPrompt])
    class ThinkSut(ThinkingMixin, FakeBaseSUT):
        pass

    @pytest.fixture
    def sut(self):
        return self.ThinkSut("sut-uid")

    def test_response_contains_reasoning(self):
        response = SUTResponse(text="reasoning</think>output")
        assert self.ThinkSut.response_contains_reasoning(response) is True

        response = SUTResponse(text="<think>reasoning</think>output")
        assert self.ThinkSut.response_contains_reasoning(response) is True

        response = SUTResponse(text="<think> only thinking")
        assert self.ThinkSut.response_contains_reasoning(response) is True

        response = SUTResponse(text="content")
        assert self.ThinkSut.response_contains_reasoning(response) is False

        response = SUTResponse(text="")
        assert self.ThinkSut.response_contains_reasoning(response) is False

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

        options = ModelOptions()
        request = sut.translate_text_prompt(prompt, options)
        assert request.request.max_tokens == None
        assert request.max_content_tokens == None

    @pytest.mark.parametrize(
        "full_text, content_text, reason_text",
        [
            ("<think>hmm</think>\n Output", "Output", "hmm"),
            (
                "<think>hmm <think> nested think> </think></think>\n Output",
                "Output",
                "hmm <think> nested think> </think>",
            ),
            ("hmm</think><think>more think</think> Output", "Output", "hmm</think><think>more think"),
            ("hmm</think>\n Output", "Output", "hmm"),
            ("<think>hmmm", "", "hmmm"),
            ("<think>", "", ""),
        ],
    )
    def test_translate_response_no_truncation(self, full_text, content_text, reason_text, sut):
        request = ReasoningRequest(
            request=FakeSUTRequest(text="", max_tokens=100), max_content_tokens=100, max_total_tokens=100
        )
        response = FakeSUTResponse(text=full_text)

        result = sut.translate_response(request, response)
        assert result.text == content_text

        request = ReasoningRequest(request=FakeSUTRequest(text=""))
        response = FakeSUTResponse(text=full_text)

        result = sut.translate_response(request, response)
        assert result.text == content_text
        assert result.reasoning == reason_text

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
        request = ReasoningRequest(
            request=FakeSUTRequest(text="", max_tokens=100), max_content_tokens=2, max_total_tokens=100
        )
        response = FakeSUTResponse(text=full_text)

        result = sut.translate_response(request, response)
        assert result.text == content_text

    def test_translate_response_warns_reasoning_over_budget(self, sut, caplog):
        request = ReasoningRequest(
            request=FakeSUTRequest(text="", max_tokens=5), max_content_tokens=2, max_total_tokens=5
        )
        response = FakeSUTResponse(text="one two three four</think> five")

        result = sut.translate_response(request, response)
        assert "reasoning likely ate into the token budget of the actual output" in caplog.text
