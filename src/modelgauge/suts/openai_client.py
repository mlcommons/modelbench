from typing import Any, Dict, List, Optional, Union

import openai
from openai import APITimeoutError, ConflictError, InternalServerError, RateLimitError
from openai import OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from modelgauge.auth.openai_compatible_secrets import (
    OpenAIApiKey,
    OpenAICompatibleApiKey,
    OpenAIOrganization,
)
from modelgauge.prompt import ChatPrompt, ChatRole, TextPrompt
from modelgauge.retry_decorator import retry
from modelgauge.secret_values import InjectSecret
from modelgauge.sut import (
    PromptResponseSUT,
    SUTOptions,
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

_SYSTEM_ROLE = "system"
_USER_ROLE = "user"
_ASSISTANT_ROLE = "assistant"
_TOOL_ROLE = "tool_call_id"

_ROLE_MAP = {
    ChatRole.user: _USER_ROLE,
    ChatRole.sut: _ASSISTANT_ROLE,
    ChatRole.system: _SYSTEM_ROLE,
}


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
    max_completion_tokens: Optional[int] = None
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
        self,
        uid: str,
        model: str,
        api_key: Optional[OpenAICompatibleApiKey] = None,
        organization: Optional[OpenAIOrganization] = None,
        base_url: Optional[str] = None,
        client: Optional[OpenAI] = None,
    ):
        super().__init__(uid)
        self.model = model
        self.api_key = api_key.value if api_key else None
        self.organization = organization.value if organization else None
        self.base_url = base_url
        self.client = client

        # key and optional org id if you're talking to openAI
        # key and base_url if you're using this client to interact with e.g. gemini on google's hardware
        assert self.client or self.api_key
        # base url or organization, not both
        if self.base_url:
            assert not self.organization
        if self.organization:
            assert not self.base_url

    def _load_client(self) -> OpenAI | None:
        if self.client:
            return self.client

        if self.base_url:
            return OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=7)
        elif self.organization:
            return OpenAI(api_key=self.api_key, organization=self.organization, max_retries=7)
        else:
            return OpenAI(api_key=self.api_key, max_retries=7)

    def translate_text_prompt(self, prompt: TextPrompt, options: SUTOptions) -> OpenAIChatRequest:
        messages = [OpenAIChatMessage(content=prompt.text, role=_USER_ROLE)]
        return self._translate_request(messages, options)

    def translate_chat_prompt(self, prompt: ChatPrompt, options: SUTOptions) -> OpenAIChatRequest:
        messages = []
        for message in prompt.messages:
            messages.append(OpenAIChatMessage(content=message.text, role=_ROLE_MAP[message.role]))
        return self._translate_request(messages, options)

    def _translate_request(self, messages: List[OpenAIChatMessage], options: SUTOptions):
        optional_kwargs: Dict[str, Any] = {}
        if options.top_logprobs is not None:
            optional_kwargs["logprobs"] = True
            optional_kwargs["top_logprobs"] = min(options.top_logprobs, 20)
        return OpenAIChatRequest(
            messages=messages,
            model=self.model,
            frequency_penalty=options.frequency_penalty,
            max_completion_tokens=options.max_tokens,
            presence_penalty=options.presence_penalty,
            stop=options.stop_sequences,
            temperature=options.temperature,
            top_p=options.top_p,
            **optional_kwargs,
        )

    @retry(transient_exceptions=[APITimeoutError, ConflictError, InternalServerError, RateLimitError])
    def evaluate(self, request: OpenAIChatRequest) -> ChatCompletion:
        if self.client is None:
            # Handle lazy init.
            self.client = self._load_client()
        try:
            return self.client.chat.completions.create(**self.request_as_dict_for_client(request))
        except openai.NotFoundError as e:
            if self.base_url:
                raise ValueError(f"404 for base URL {self.base_url}") from e
            else:
                raise
        except openai.APIConnectionError as e:
            if self.base_url:
                raise ValueError(f"Couldn't connect to base URL {self.base_url}") from e
            else:
                raise

    def request_as_dict_for_client(self, request: OpenAIChatRequest) -> dict[str, Any]:
        return request.model_dump(exclude_none=True)

    def translate_response(self, request: OpenAIChatRequest, response: ChatCompletion) -> SUTResponse:
        assert len(response.choices) == 1, f"Expected a single response message, got {len(response.choices)}."
        choice = response.choices[0]
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
                    top_tokens.append(TokenProbability(token=top.token, logprob=top.logprob))
                logprobs.append(TopTokens(top_tokens=top_tokens))
        assert text is not None
        return SUTResponse(text=text, top_logprobs=logprobs)


SUTS.register(
    OpenAIChat,
    "gpt-3.5-turbo",
    "gpt-3.5-turbo",
    InjectSecret(OpenAIApiKey),
    InjectSecret(OpenAIOrganization),
)

SUTS.register(
    OpenAIChat,
    "gpt-4o",
    "gpt-4o",
    InjectSecret(OpenAIApiKey),
    InjectSecret(OpenAIOrganization),
)

SUTS.register(
    OpenAIChat,
    "gpt-4o-20250508",
    "gpt-4o",
    InjectSecret(OpenAIApiKey),
    InjectSecret(OpenAIOrganization),
)

SUTS.register(
    OpenAIChat,
    "gpt-4o-mini",
    "gpt-4o-mini",
    InjectSecret(OpenAIApiKey),
    InjectSecret(OpenAIOrganization),
)
