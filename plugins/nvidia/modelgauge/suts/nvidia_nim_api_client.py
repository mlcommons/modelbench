from typing import Any, Dict, List, Optional, Union

from openai import OpenAI
from openai import APITimeoutError, ConflictError, InternalServerError, RateLimitError
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from modelgauge.prompt import ChatPrompt, ChatRole, SUTOptions, TextPrompt
from modelgauge.retry_decorator import retry
from modelgauge.secret_values import (
    InjectSecret,
    RequiredSecret,
    SecretDescription,
)
from modelgauge.sut import (
    PromptResponseSUT,
    SUTCompletion,
    SUTResponse,
)
from modelgauge.sut_capabilities import (
    AcceptsChatPrompt,
    AcceptsTextPrompt,
)
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS

_SYSTEM_ROLE = "system"
_USER_ROLE = "user"
_ASSISTANT_ROLE = "assistant"
_TOOL_ROLE = "tool_call_id"

_ROLE_MAP = {
    ChatRole.user: _USER_ROLE,
    ChatRole.sut: _ASSISTANT_ROLE,
    ChatRole.system: _SYSTEM_ROLE,
}


class NvidiaNIMApiKey(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="nvidia-nim-api",
            key="api_key",
            instructions="See https://build.nvidia.com/",
        )


class OpenAIChatMessage(BaseModel):
    content: str
    role: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None


class OpenAIChatRequest(BaseModel):
    messages: List[OpenAIChatMessage]
    model: str
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[bool] = None
    max_tokens: Optional[int] = 256
    n: Optional[int] = 1
    presence_penalty: Optional[float] = None
    response_format: Optional[Dict] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    tools: Optional[List] = None
    tool_choice: Optional[Union[str, Dict]] = None
    user: Optional[str] = None


@modelgauge_sut(
    capabilities=[
        AcceptsTextPrompt,
        AcceptsChatPrompt,
    ]
)
class NvidiaNIMApiClient(PromptResponseSUT[OpenAIChatRequest, ChatCompletion]):
    """
    Documented at https://https://docs.api.nvidia.com/
    """

    def __init__(self, uid: str, model: str, api_key: NvidiaNIMApiKey):
        super().__init__(uid)
        self.model = model
        self.client: Optional[OpenAI] = None
        self.api_key = api_key.value

    def _load_client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key, base_url="https://integrate.api.nvidia.com/v1")

    def translate_text_prompt(self, prompt: TextPrompt) -> OpenAIChatRequest:
        messages = [OpenAIChatMessage(content=prompt.text, role=_USER_ROLE)]
        return self._translate_request(messages, prompt.options)

    def translate_chat_prompt(self, prompt: ChatPrompt) -> OpenAIChatRequest:
        messages = []
        for message in prompt.messages:
            messages.append(OpenAIChatMessage(content=message.text, role=_ROLE_MAP[message.role]))
        return self._translate_request(messages, prompt.options)

    def _translate_request(self, messages: List[OpenAIChatMessage], options: SUTOptions):
        optional_kwargs: Dict[str, Any] = {}
        return OpenAIChatRequest(
            messages=messages,
            model=self.model,
            frequency_penalty=options.frequency_penalty,
            max_tokens=options.max_tokens,
            n=options.num_completions,
            presence_penalty=options.presence_penalty,
            stop=options.stop_sequences,
            top_p=options.top_p,
            **optional_kwargs,
        )

    @retry(transient_exceptions=[APITimeoutError, ConflictError, InternalServerError, RateLimitError])
    def evaluate(self, request: OpenAIChatRequest) -> ChatCompletion:
        if self.client is None:
            # Handle lazy init.
            self.client = self._load_client()
        request_dict = request.model_dump(exclude_none=True)
        return self.client.chat.completions.create(**request_dict)

    def translate_response(self, request: OpenAIChatRequest, response: ChatCompletion) -> SUTResponse:
        completions = []
        for choice in response.choices:
            text = choice.message.content
            if text is None:
                text = ""
            completions.append(SUTCompletion(text=text))
        return SUTResponse(completions=completions)


SUTS.register(
    NvidiaNIMApiClient,
    "nvidia-llama-3.1-nemotron-70b-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    InjectSecret(NvidiaNIMApiKey),
)


SUTS.register(
    NvidiaNIMApiClient,
    "nvidia-nemotron-4-340b-instruct",
    "nvidia/nemotron-4-340b-instruct",
    InjectSecret(NvidiaNIMApiKey),
)

SUTS.register(
    NvidiaNIMApiClient,
    "nvidia-mistral-nemo-minitron-8b-8k-instruct",
    "nvidia/mistral-nemo-minitron-8b-8k-instruct",
    InjectSecret(NvidiaNIMApiKey),
)

SUTS.register(
    NvidiaNIMApiClient,
    "nvidia-nemotron-mini-4b-instruct",
    "nvidia/nemotron-mini-4b-instruct",
    InjectSecret(NvidiaNIMApiKey),
)
