from typing import Any
from pydantic import BaseModel

from modelgauge.log_config import get_logger
from modelgauge.model_options import ModelOptions
from modelgauge.prompt import TextPrompt
from modelgauge.sut import SUTResponse, PromptResponseSUT
from modelgauge.tokenizer import GeneralTokenizer

logger = get_logger(__name__)


class ReasoningRequest(BaseModel):
    request: Any
    max_content_tokens: int | None = None  # Number of tokens allowed for content (excluding thinking text).


class ThinkingMixin(PromptResponseSUT):
    """
    A mixin for SUTs that parses out thinking text from the output.

    Reasoning is expected to be seperated from  content by a </think> tags.
    If max_total_output_tokens is set in ModelOptions, that value will be used in the model call
    and the content text will be truncated to max_tokens.
    Otherwise, max_tokens is used in the model call and everything after </think> is returned as content.
    """

    def __init__(self, uid, *args, **kwargs):
        super().__init__(uid, *args, **kwargs)
        self.tokenizer = GeneralTokenizer()

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
        )

    def evaluate(self, request: ReasoningRequest) -> Any:
        return super().evaluate(request.request)  # type: ignore

    def translate_response(self, request: ReasoningRequest, response: Any) -> SUTResponse:
        text = super().translate_response(request.request, response).text  # type: ignore

        think_close = text.find("</think>")
        if think_close == -1:
            # no closing tag: everything is thinking text
            return SUTResponse(text="")

        # Warn if reasoning was so large that remaining content had less than max_content_tokens available.
        reasoning = text[: think_close + len("</think>")].strip()
        content = text[think_close + len("</think>") :].strip()

        reasoning_tokens = self.tokenizer.count_tokens(reasoning)
        content_tokens = self.tokenizer.count_tokens(content)
        reasoning_budget = request.request.max_tokens - request.max_content_tokens

        if reasoning_tokens >= reasoning_budget and content_tokens >= request.max_content_tokens:
            logger.warning(
                f"SUT {self.uid} reasoning likely ate into the token budget of the actual output. Consider increasing max_total_output_tokens."
            )
        # Truncate content
        content_text = self.tokenizer.truncate(content, request.max_content_tokens)

        return SUTResponse(text=content_text)
