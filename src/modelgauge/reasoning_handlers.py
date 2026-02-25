from abc import abstractmethod
from typing import Any

from airrlogger.log_config import get_logger
from pydantic import BaseModel

from modelgauge.general import get_concrete_subclasses
from modelgauge.model_options import ModelOptions
from modelgauge.prompt import TextPrompt
from modelgauge.sut import PromptResponseSUT, SUTResponse
from modelgauge.tokenizer import GeneralTokenizer

logger = get_logger(__name__)


class ReasoningRequest(BaseModel):
    request: Any  # Request that is actually sent to the model.
    max_content_tokens: int | None = None  # Number of tokens allowed for content (excluding thinking text).
    max_total_tokens: int | None = None  # Total number of tokens allowed (thinking + content).


class ReasoningSUT(PromptResponseSUT):
    @staticmethod
    def _get_concrete_reasoning_suts() -> set[type["ReasoningSUT"]]:
        return get_concrete_subclasses(ReasoningSUT)

    @staticmethod
    def find_match(sut: PromptResponseSUT) -> type["ReasoningSUT"] | None:
        reasoning_suts = ReasoningSUT._get_concrete_reasoning_suts()
        for rs in reasoning_suts:
            if rs.sut_matches(sut):
                return rs
        return None

    @classmethod
    def sut_matches(cls, sut) -> bool:
        request = sut.translate_text_prompt(
            TextPrompt(text="If I have 2 apples and give 1 to my friend, how many apples do I have left?"),
            options=ModelOptions(max_tokens=1000),
        )
        raw_response = sut.evaluate(request)
        response = sut.translate_response(request, raw_response)
        return cls.response_contains_reasoning(response)

    @classmethod
    @abstractmethod
    def response_contains_reasoning(cls, response: SUTResponse) -> bool:
        pass


class ThinkingMixin(ReasoningSUT):
    """
    A mixin for SUTs that parses out thinking text from the output.

    The output is expected to be in the form: {reasoning text}</think>{content text}.
    If max_total_output_tokens is set in ModelOptions, that value will be used in the model call and the content text will be truncated to max_tokens.
    Otherwise, max_tokens is used in the model call and everything after </think> is returned as content.

    Reasoning should be enabled by the model by default. This mixin does not request reasoning be enabled (yet).
    """

    OPEN_TAG = "<think>"  # Optional.
    CLOSE_TAG = "</think>"  # Tag that separates reasoning from content.

    def __init__(self, uid, *args, **kwargs):
        super().__init__(uid, *args, **kwargs)
        self.tokenizer = GeneralTokenizer()

    @classmethod
    def response_contains_reasoning(cls, response: SUTResponse) -> bool:
        return cls.OPEN_TAG in response.text or cls.CLOSE_TAG in response.text

    def translate_text_prompt(self, prompt: TextPrompt, options: ModelOptions) -> ReasoningRequest:
        max_total_tokens = options.max_total_output_tokens
        if max_total_tokens is None:
            max_total_tokens = options.max_tokens
        max_content_tokens = options.max_tokens

        # Replace max_tokens in raw request with the max total tokens.
        options.max_tokens = max_total_tokens
        request = super().translate_text_prompt(prompt, options)
        return ReasoningRequest(
            request=request,
            max_content_tokens=max_content_tokens,
            max_total_tokens=max_total_tokens,
        )

    def evaluate(self, request: ReasoningRequest) -> Any:
        return super().evaluate(request.request)  # type: ignore

    def translate_response(self, request: ReasoningRequest, response: Any) -> SUTResponse:
        text = super().translate_response(request.request, response).text  # type: ignore

        think_close = text.find(self.CLOSE_TAG)
        if think_close == -1:
            # no closing tag: everything is thinking text
            return SUTResponse(text="", reasoning=self.trim_tokens(text))

        reasoning = text[: think_close + len(self.CLOSE_TAG)].strip()
        content = text[think_close + len(self.CLOSE_TAG) :].strip()
        self.warn_edge_cases(content, reasoning, request)

        reasoning = self.trim_tokens(reasoning)

        # Truncate content
        if request.max_content_tokens is not None:
            content = self.tokenizer.truncate(content, request.max_content_tokens)
        return SUTResponse(text=content, reasoning=reasoning)

    def warn_edge_cases(self, content, reasoning, request):
        if request.max_total_tokens is None:
            return
        reasoning_tokens = self.tokenizer.count_tokens(reasoning)
        content_tokens = self.tokenizer.count_tokens(content)
        reasoning_budget = request.request.max_tokens - request.max_content_tokens

        if reasoning_tokens >= reasoning_budget and content_tokens + reasoning_tokens >= request.max_total_tokens:
            logger.warning(
                f"SUT {self.uid} reasoning likely ate into the token budget of the actual output. Consider increasing max_total_output_tokens."
            )

    def trim_tokens(self, text: str) -> str:
        if text.startswith(self.OPEN_TAG):
            text = text[len(self.OPEN_TAG) :]
        if text.endswith(self.CLOSE_TAG):
            text = text[: -len(self.CLOSE_TAG)]
        return text
