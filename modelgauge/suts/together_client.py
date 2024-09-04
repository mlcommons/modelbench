from typing import Any, List, Optional

import requests  # type:ignore
from pydantic import BaseModel, Field
from requests.adapters import HTTPAdapter, Retry  # type:ignore

from modelgauge.general import APIException
from modelgauge.prompt import ChatPrompt, ChatRole, SUTOptions, TextPrompt
from modelgauge.prompt_formatting import format_chat
from modelgauge.secret_values import InjectSecret, RequiredSecret, SecretDescription
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


class TogetherApiKey(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="together",
            key="api_key",
            instructions="See https://api.together.xyz/settings/api-keys",
        )


_SYSTEM_ROLE = "system"
_USER_ROLE = "user"
_ASSISTANT_ROLE = "assistant"

_ROLE_MAP = {
    ChatRole.user: _USER_ROLE,
    ChatRole.sut: _ASSISTANT_ROLE,
    ChatRole.system: _SYSTEM_ROLE,
}


def _retrying_post(url, headers, json_payload):
    """HTTP Post with retry behavior."""
    session = requests.Session()
    retries = Retry(
        total=6,
        backoff_factor=2,
        status_forcelist=[
            408,  # Request Timeout
            421,  # Misdirected Request
            423,  # Locked
            424,  # Failed Dependency
            425,  # Too Early
            429,  # Too Many Requests
        ]
        + list(range(500, 599)),  # Add all 5XX.
        allowed_methods=["POST"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    response = None
    try:
        response = session.post(url, headers=headers, json=json_payload, timeout=120)
        return response
    except Exception as e:
        raise Exception(
            f"Exception calling {url} with {json_payload}. Response {response.text if response else response}"
        ) from e


class TogetherCompletionsRequest(BaseModel):
    # https://docs.together.ai/reference/completions
    model: str
    prompt: str
    max_tokens: int
    stop: Optional[List[str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    n: Optional[int] = None  # How many completions.
    logprobs: Optional[int] = None


class TogetherLogProbs(BaseModel):
    tokens: List[str]
    token_logprobs: List[float]


class TogetherCompletionsResponse(BaseModel):
    # https://docs.together.ai/reference/completions

    class Choice(BaseModel):
        text: str
        logprobs: Optional[TogetherLogProbs] = None

    class Usage(BaseModel):
        prompt_tokens: int
        completion_tokens: int
        total_tokens: int

    id: str
    choices: List[Choice]
    usage: Usage
    created: int
    model: str
    object: str


@modelgauge_sut(
    capabilities=[
        AcceptsTextPrompt,
        AcceptsChatPrompt,
        ProducesPerTokenLogProbabilities,
    ]
)
class TogetherCompletionsSUT(
    PromptResponseSUT[TogetherCompletionsRequest, TogetherCompletionsResponse]
):
    _URL = "https://api.together.xyz/v1/completions"

    def __init__(self, uid: str, model, api_key: TogetherApiKey):
        super().__init__(uid)
        self.model = model
        self.api_key = api_key.value

    def translate_text_prompt(self, prompt: TextPrompt) -> TogetherCompletionsRequest:
        return self._translate_request(prompt.text, prompt.options)

    def translate_chat_prompt(self, prompt: ChatPrompt) -> TogetherCompletionsRequest:
        return self._translate_request(
            format_chat(prompt, user_role=_USER_ROLE, sut_role=_ASSISTANT_ROLE),
            prompt.options,
        )

    def _translate_request(self, text, options):
        return TogetherCompletionsRequest(
            model=self.model,
            prompt=text,
            max_tokens=options.max_tokens,
            stop=options.stop_sequences,
            temperature=options.temperature,
            top_p=options.top_p,
            top_k=options.top_k_per_token,
            repetition_penalty=options.frequency_penalty,
            n=options.num_completions,
            logprobs=options.top_logprobs,
        )

    def evaluate(
        self, request: TogetherCompletionsRequest
    ) -> TogetherCompletionsResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        as_json = request.model_dump(exclude_none=True)
        response = _retrying_post(self._URL, headers, as_json)
        if not response.status_code == 200:
            raise APIException(
                f"Unexpected API failure ({response.status_code}): {response.text}"
            )
        return TogetherCompletionsResponse.model_validate(response.json(), strict=True)

    def translate_response(
        self, request: TogetherCompletionsRequest, response: TogetherCompletionsResponse
    ) -> SUTResponse:
        sut_completions = []
        for choice in response.choices:
            assert choice.text is not None
            logprobs: Optional[List[TopTokens]] = None
            if request.logprobs:
                logprobs = []
                assert (
                    choice.logprobs is not None
                ), "Expected logprobs, but not returned."
                for token, logprob in zip(
                    choice.logprobs.tokens, choice.logprobs.token_logprobs
                ):
                    # Together only returns 1 logprob/token
                    logprobs.append(
                        TopTokens(
                            top_tokens=[TokenProbability(token=token, logprob=logprob)]
                        )
                    )
            sut_completions.append(
                SUTCompletion(text=choice.text, top_logprobs=logprobs)
            )
        return SUTResponse(completions=sut_completions)


class TogetherChatRequest(BaseModel):
    # https://docs.together.ai/reference/chat-completions
    class Message(BaseModel):
        role: str
        content: str

    model: str
    messages: List[Message]
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    n: Optional[int] = None
    logprobs: Optional[int] = None


class TogetherChatResponse(BaseModel):
    # https://docs.together.ai/reference/chat-completions
    class Choice(BaseModel):
        class Message(BaseModel):
            role: str
            content: str

        message: Message
        logprobs: Optional[TogetherLogProbs] = None

    class Usage(BaseModel):
        prompt_tokens: int
        completion_tokens: int
        total_tokens: int

    id: str
    choices: List[Choice]
    usage: Usage
    created: int
    model: str
    object: str


@modelgauge_sut(
    capabilities=[
        AcceptsTextPrompt,
        AcceptsChatPrompt,
        ProducesPerTokenLogProbabilities,
    ]
)
class TogetherChatSUT(PromptResponseSUT[TogetherChatRequest, TogetherChatResponse]):
    _URL = "https://api.together.xyz/v1/chat/completions"

    def __init__(self, uid: str, model, api_key: TogetherApiKey):
        super().__init__(uid)
        self.model = model
        self.api_key = api_key.value

    def translate_text_prompt(self, prompt: TextPrompt) -> TogetherChatRequest:
        return self._translate_request(
            [TogetherChatRequest.Message(content=prompt.text, role=_USER_ROLE)],
            prompt.options,
        )

    def translate_chat_prompt(self, prompt: ChatPrompt) -> TogetherChatRequest:
        messages = []
        for message in prompt.messages:
            messages.append(
                TogetherChatRequest.Message(
                    content=message.text, role=_ROLE_MAP[message.role]
                )
            )
        return self._translate_request(messages, prompt.options)

    def _translate_request(
        self, messages: List[TogetherChatRequest.Message], options: SUTOptions
    ):
        return TogetherChatRequest(
            model=self.model,
            messages=messages,
            max_tokens=options.max_tokens,
            stop=options.stop_sequences,
            temperature=options.temperature,
            top_p=options.top_p,
            top_k=options.top_k_per_token,
            repetition_penalty=options.frequency_penalty,
            n=options.num_completions,
            logprobs=options.top_logprobs,
        )

    def evaluate(self, request: TogetherChatRequest) -> TogetherChatResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        as_json = request.model_dump(exclude_none=True)
        response = _retrying_post(self._URL, headers, as_json)
        if not response.status_code == 200:
            raise APIException(
                f"Unexpected API failure ({response.status_code}): {response.text}"
            )
        return TogetherChatResponse.model_validate(response.json(), strict=True)

    def translate_response(
        self, request: TogetherChatRequest, response: TogetherChatResponse
    ) -> SUTResponse:
        sut_completions = []
        for choice in response.choices:
            text = choice.message.content
            assert text is not None
            logprobs: Optional[List[TopTokens]] = None
            if request.logprobs:
                logprobs = []
                assert (
                    choice.logprobs is not None
                ), "Expected logprobs, but not returned."
                for token, logprob in zip(
                    choice.logprobs.tokens, choice.logprobs.token_logprobs
                ):
                    # Together only returns 1 logprob/token
                    logprobs.append(
                        TopTokens(
                            top_tokens=[TokenProbability(token=token, logprob=logprob)]
                        )
                    )

            sut_completions.append(SUTCompletion(text=text, top_logprobs=logprobs))
        return SUTResponse(completions=sut_completions)


class TogetherInferenceRequest(BaseModel):
    # https://docs.together.ai/reference/inference
    model: str
    # prompt is documented as required, but you can pass messages instead,
    # which is not documented.
    prompt: Optional[str] = None
    messages: Optional[List[TogetherChatRequest.Message]] = None
    max_tokens: int
    stop: Optional[List[str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    safety_model: Optional[str] = None
    n: Optional[int] = None
    logprobs: Optional[int] = None


class TogetherInferenceResponse(BaseModel):
    class Args(BaseModel):
        model: str
        prompt: Optional[str] = None
        temperature: Optional[float] = None
        top_p: Optional[float] = None
        top_k: Optional[float] = None
        max_tokens: int

    status: str
    prompt: List[str]
    model: str
    # Pydantic uses "model_" as the prefix for its methods, so renaming
    # here to get out of the way.
    owner: str = Field(alias="model_owner")
    tags: Optional[Any] = None
    num_returns: int
    args: Args
    subjobs: List

    class Output(BaseModel):
        class Choice(BaseModel):
            finish_reason: str
            index: Optional[int] = None
            text: str
            tokens: Optional[List[str]] = None
            token_logprobs: Optional[List[float]] = None

        choices: List[Choice]
        raw_compute_time: Optional[float] = None
        result_type: str

    output: Output


@modelgauge_sut(
    capabilities=[
        AcceptsTextPrompt,
        AcceptsChatPrompt,
        ProducesPerTokenLogProbabilities,
    ]
)
class TogetherInferenceSUT(
    PromptResponseSUT[TogetherInferenceRequest, TogetherInferenceResponse]
):
    _URL = "https://api.together.xyz/inference"

    def __init__(self, uid: str, model, api_key: TogetherApiKey):
        super().__init__(uid)
        self.model = model
        self.api_key = api_key.value

    def translate_text_prompt(self, prompt: TextPrompt) -> TogetherInferenceRequest:
        return self._translate_request(prompt.text, prompt.options)

    def translate_chat_prompt(self, prompt: ChatPrompt) -> TogetherInferenceRequest:
        return self._translate_request(
            format_chat(prompt, user_role=_USER_ROLE, sut_role=_ASSISTANT_ROLE),
            prompt.options,
        )

    def _translate_request(self, text: str, options: SUTOptions):
        return TogetherInferenceRequest(
            model=self.model,
            prompt=text,
            max_tokens=options.max_tokens,
            stop=options.stop_sequences,
            temperature=options.temperature,
            top_p=options.top_p,
            top_k=options.top_k_per_token,
            repetition_penalty=options.frequency_penalty,
            n=options.num_completions,
            logprobs=options.top_logprobs,
        )

    def evaluate(self, request: TogetherInferenceRequest) -> TogetherInferenceResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        as_json = request.model_dump(exclude_none=True)
        response = _retrying_post(self._URL, headers, as_json)
        if not response.status_code == 200:
            raise APIException(
                f"Unexpected API failure ({response.status_code}): {response.text}"
            )
        return TogetherInferenceResponse(**response.json())

    def translate_response(
        self, request: TogetherInferenceRequest, response: TogetherInferenceResponse
    ) -> SUTResponse:
        sut_completions = []
        for choice in response.output.choices:
            assert choice.text is not None
            logprobs: Optional[List[TopTokens]] = None
            if request.logprobs:
                logprobs = []
                assert (
                    choice.tokens is not None and choice.token_logprobs is not None
                ), "Expected logprobs, but not returned."
                for token, logprob in zip(choice.tokens, choice.token_logprobs):
                    logprobs.append(
                        TopTokens(
                            top_tokens=[TokenProbability(token=token, logprob=logprob)]
                        )
                    )
            sut_completions.append(
                SUTCompletion(text=choice.text, top_logprobs=logprobs)
            )
        return SUTResponse(completions=sut_completions)


LANGUAGE_MODELS: dict[str, str] = {
    # This was deprecated around 2024-08-29
    # You may add some once they are available.
    # Leaving this empty dict in the code as an example to get you started.
    # "llama-2-7b": "meta-llama/Llama-2-7b-hf",
}
for uid, model_name in LANGUAGE_MODELS.items():
    SUTS.register(TogetherCompletionsSUT, uid, model_name, InjectSecret(TogetherApiKey))

# Chat
CHAT_MODELS = {
    "llama-2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "llama-3-70b-chat": "meta-llama/Llama-3-70b-chat-hf",
    "llama-2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
    "llama-3-70b-chat-hf": "meta-llama/Llama-3-70b-chat-hf",
    "Mistral-7B-Instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
    "Mixtral-8x7B-Instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistral-8x22b-instruct": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "deepseek-llm-67b-chat": "deepseek-ai/deepseek-llm-67b-chat",
    "StripedHyena-Nous-7B": "togethercomputer/StripedHyena-Nous-7B",
    "Qwen1.5-72B-Chat": "Qwen/Qwen1.5-72B-Chat",
}
for uid, model_name in CHAT_MODELS.items():
    SUTS.register(TogetherChatSUT, uid, model_name, InjectSecret(TogetherApiKey))
