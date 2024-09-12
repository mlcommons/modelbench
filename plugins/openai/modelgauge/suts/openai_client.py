from modelgauge.prompt import ChatPrompt, ChatRole, SUTOptions, TextPrompt
from modelgauge.secret_values import (
    InjectSecret,
    OptionalSecret,
    RequiredSecret,
    SecretDescription,
)
from modelgauge.sut import (
    PromptResponseSUT,
    SUTCompletion,
    SUTResponse,
    TokenProbability,
    TopTokens,
)
from modelgauge.sut_capabilities import (
    AcceptsChatPrompt,
    AcceptsTextPrompt,
    ProducesPerTokenLogProbabilities,
)
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS
from openai import OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union

_SYSTEM_ROLE = "system"
_USER_ROLE = "user"
_ASSISTANT_ROLE = "assistant"
_TOOL_ROLE = "tool_call_id"

_ROLE_MAP = {
    ChatRole.user: _USER_ROLE,
    ChatRole.sut: _ASSISTANT_ROLE,
    ChatRole.system: _SYSTEM_ROLE,
}


class OpenAIApiKey(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="openai",
            key="api_key",
            instructions="See https://platform.openai.com/api-keys",
        )


class OpenAIOrgId(OptionalSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="openai",
            key="org_id",
            instructions="See https://platform.openai.com/account/organization",
        )


class OpenAIChatMessage(BaseModel):
    content: str
    role: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None


class OpenAIChatRequest(BaseModel):
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: List[OpenAIChatMessage]
    model: str
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[bool] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    # How many chat completion choices to generate for each input message.
    n: Optional[int] = None
    presence_penalty: Optional[float] = None
    response_format: Optional[Dict] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[List] = None
    tool_choice: Optional[Union[str, Dict]] = None
    user: Optional[str] = None


@modelgauge_sut(
    capabilities=[
        AcceptsTextPrompt,
        AcceptsChatPrompt,
        ProducesPerTokenLogProbabilities,
    ]
)
class OpenAIChat(PromptResponseSUT[OpenAIChatRequest, ChatCompletion]):
    """
    Documented at https://platform.openai.com/docs/api-reference/chat/create
    """

    def __init__(
        self, uid: str, model: str, api_key: OpenAIApiKey, org_id: OpenAIOrgId
    ):
        super().__init__(uid)
        self.model = model
        self.client: Optional[OpenAI] = None
        self.api_key = api_key.value
        self.org_id = org_id.value

    def _load_client(self) -> OpenAI:
        return OpenAI(
            api_key=self.api_key,
            organization=self.org_id,
        )

    def translate_text_prompt(self, prompt: TextPrompt) -> OpenAIChatRequest:
        messages = [OpenAIChatMessage(content=prompt.text, role=_USER_ROLE)]
        return self._translate_request(messages, prompt.options)

    def translate_chat_prompt(self, prompt: ChatPrompt) -> OpenAIChatRequest:
        messages = []
        for message in prompt.messages:
            messages.append(
                OpenAIChatMessage(content=message.text, role=_ROLE_MAP[message.role])
            )
        return self._translate_request(messages, prompt.options)

    def _translate_request(
        self, messages: List[OpenAIChatMessage], options: SUTOptions
    ):
        optional_kwargs: Dict[str, Any] = {}
        if options.top_logprobs is not None:
            optional_kwargs["logprobs"] = True
            optional_kwargs["top_logprobs"] = min(options.top_logprobs, 20)
        return OpenAIChatRequest(
            messages=messages,
            model=self.model,
            frequency_penalty=options.frequency_penalty,
            max_tokens=options.max_tokens,
            n=options.num_completions,
            presence_penalty=options.presence_penalty,
            stop=options.stop_sequences,
            temperature=options.temperature,
            top_p=options.top_p,
            **optional_kwargs,
        )

    def evaluate(self, request: OpenAIChatRequest) -> ChatCompletion:
        if self.client is None:
            # Handle lazy init.
            self.client = self._load_client()
        request_dict = request.model_dump(exclude_none=True)
        return self.client.chat.completions.create(**request_dict)

    def translate_response(
        self, request: OpenAIChatRequest, response: ChatCompletion
    ) -> SUTResponse:
        completions = []
        for choice in response.choices:
            text = choice.message.content
            logprobs: Optional[List[TopTokens]] = None
            if request.logprobs:
                logprobs = []
                assert (
                    choice.logprobs is not None and choice.logprobs.content is not None
                ), "Expected logprobs, but not returned."
                for token_content in choice.logprobs.content:
                    top_tokens: List[TokenProbability] = []
                    for top in token_content.top_logprobs:
                        top_tokens.append(
                            TokenProbability(token=top.token, logprob=top.logprob)
                        )
                    logprobs.append(TopTokens(top_tokens=top_tokens))
            assert text is not None
            completions.append(SUTCompletion(text=text, top_logprobs=logprobs))
        return SUTResponse(completions=completions)


SUTS.register(
    OpenAIChat,
    "gpt-3.5-turbo",
    "gpt-3.5-turbo",
    InjectSecret(OpenAIApiKey),
    InjectSecret(OpenAIOrgId),
)
