from random import random
from time import sleep
from typing import List, Optional

import anthropic
from anthropic import Anthropic
from anthropic.types import TextBlock
from anthropic.types.message import Message as AnthropicMessage
from pydantic import BaseModel

from airrlogger.log_config import get_logger

from modelgauge.general import APIException
from modelgauge.prompt import ChatRole, TextPrompt
from modelgauge.secret_values import InjectSecret, RequiredSecret, SecretDescription
from modelgauge.sut import REFUSAL_RESPONSE, PromptResponseSUT, SUTResponse
from modelgauge.model_options import ModelOptions
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS
from modelgauge.suts.openai_client import OpenAIChatMessage, _ROLE_MAP

logger = get_logger(__name__)


class AnthropicApiKey(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="anthropic",
            key="api_key",
            instructions="You can create api keys at https://console.anthropic.com/settings/keys.",
        )


class AnthropicRequest(BaseModel):
    # Largely follows the OpenAI standard, but Anthropic does not support all the same options as OpenAI.
    model: str
    messages: List[OpenAIChatMessage]
    # Generation options:
    max_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None


@modelgauge_sut(capabilities=[AcceptsTextPrompt])
class AnthropicSUT(PromptResponseSUT):
    # Claude Opus 5 uses adaptive thinking by default, and max_tokens covers
    # thinking plus visible response text. The generic 20-token readiness budget
    # can therefore be exhausted before a TextBlock is produced.
    READINESS_CHECK_MAX_TOKENS = 1024

    def __init__(self, uid: str, model: str, api_key: AnthropicApiKey):
        super().__init__(uid)
        self.model = model
        self.api_key = api_key.value
        self.client: Optional[Anthropic] = None

    def _load_client(self) -> Anthropic:
        return Anthropic(
            api_key=self.api_key,
            max_retries=7,
        )

    @property
    def accepts_temperature(self) -> bool:
        if "claude" not in self.model:
            return True
        try:
            # Names look like claude-opus-4-6 or claude-3-5-sonnet-20241022.
            parts = self.model.split("-")[1:]
            if parts and not parts[0].isdigit():
                parts = parts[1:]  # skip family name (opus, sonnet, haiku)
            version_nums = []
            for part in parts:
                if not part.isdigit() or len(part) >= 8:  # skip YYYYMMDD dates
                    break
                version_nums.append(int(part))
            if not version_nums:
                return True
            version = tuple(version_nums)
        except (IndexError, ValueError):
            return True
        return version <= (4, 6)

    def translate_text_prompt(self, prompt: TextPrompt, options: ModelOptions) -> AnthropicRequest:
        optional_kwargs = {}
        if not self.accepts_temperature and options.temperature is not None:
            logger.warning(f"Temperature is not supported for model {self.model}, ignoring temperature.")
        elif options.temperature is not None:
            optional_kwargs["temperature"] = options.temperature
        messages = [OpenAIChatMessage(content=prompt.text, role=_ROLE_MAP[ChatRole.user])]
        return AnthropicRequest(
            model=self.model,
            messages=messages,
            max_tokens=options.max_tokens,
            stop_sequences=options.stop_sequences,
            top_k=options.top_k_per_token,
            top_p=options.top_p,
            **optional_kwargs,
        )

    def evaluate(self, request: AnthropicRequest) -> AnthropicMessage:
        if self.client is None:
            # Lazy load the client.
            self.client = self._load_client()
        request_dict = request.model_dump(exclude_none=True)
        try:
            return self.client.messages.create(**request_dict)
        except anthropic.RateLimitError:
            sleep(60 * random())  # anthropic uses 1-minute buckets
            return self.evaluate(request)
        except Exception as e:
            raise APIException(f"Error calling Anthropic API: {e}")

    def translate_response(self, request: AnthropicRequest, response: AnthropicMessage) -> SUTResponse:
        if response.stop_reason == "refusal" or not response.content:
            return SUTResponse(text=REFUSAL_RESPONSE)

        text_blocks = [block for block in response.content if isinstance(block, TextBlock)]
        assert len(text_blocks) == 1, f"Expected a single text block in the response, got {len(text_blocks)}."
        return SUTResponse(text=text_blocks[0].text)


ANTHROPIC_SECRET = InjectSecret(AnthropicApiKey)

for model in ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-7-sonnet-20250219"]:
    # UID is the model name.
    SUTS.register(AnthropicSUT, model, model, ANTHROPIC_SECRET)
