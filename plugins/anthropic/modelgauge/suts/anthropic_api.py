from anthropic import Anthropic
from anthropic.types import TextBlock
from anthropic.types.message import Message as AnthropicMessage
from pydantic import BaseModel
from typing import List, Optional

from modelgauge.general import APIException
from modelgauge.prompt import ChatRole, TextPrompt
from modelgauge.secret_values import InjectSecret, RequiredSecret, SecretDescription
from modelgauge.sut import PromptResponseSUT, SUTCompletion, SUTResponse
from modelgauge.suts.openai_client import OpenAIChatMessage, _ROLE_MAP
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS


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
class AnthropicSUT(PromptResponseSUT[AnthropicRequest, AnthropicMessage]):
    def __init__(self, uid: str, model: str, api_key: AnthropicApiKey):
        super().__init__(uid)
        self.model = model
        self.api_key = api_key.value
        self.client: Optional[Anthropic] = None

    def _load_client(self) -> Anthropic:
        return Anthropic(api_key=self.api_key)

    def translate_text_prompt(self, prompt: TextPrompt) -> AnthropicRequest:
        messages = [OpenAIChatMessage(content=prompt.text, role=_ROLE_MAP[ChatRole.user])]
        return AnthropicRequest(
            model=self.model,
            messages=messages,
            max_tokens=prompt.options.max_tokens,
            stop_sequences=prompt.options.stop_sequences,
            temperature=prompt.options.temperature,
            top_k=prompt.options.top_k_per_token,
            top_p=prompt.options.top_p,
        )

    def evaluate(self, request: AnthropicRequest) -> AnthropicMessage:
        if self.client is None:
            # Lazy load the client.
            self.client = self._load_client()
        request_dict = request.model_dump(exclude_none=True)
        try:
            return self.client.messages.create(**request_dict)
        except Exception as e:
            raise APIException(f"Error calling Anthropic API: {e}")

    def translate_response(self, request: AnthropicRequest, response: AnthropicMessage) -> SUTResponse:
        completions = []
        for text_block in response.content:
            if not isinstance(text_block, TextBlock):
                raise APIException(f"Expected TextBlock with attribute 'text', instead received {text_block}")
            completions.append(SUTCompletion(text=text_block.text))

        return SUTResponse(completions=completions)


ANTHROPIC_SECRET = InjectSecret(AnthropicApiKey)

# TODO: Add claude 3.5 Haiku when it comes out later this month
#  https://docs.anthropic.com/en/docs/about-claude/models#model-names
model_names = ["claude-3-5-sonnet-20241022"]
for model in model_names:
    # UID is the model name.
    SUTS.register(AnthropicSUT, model, model, ANTHROPIC_SECRET)
