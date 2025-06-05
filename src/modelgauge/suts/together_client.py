import logging
import time
from typing import Any, List, Optional

import requests  # type:ignore
from pydantic import BaseModel, Field
from requests.adapters import HTTPAdapter, Retry  # type:ignore

from modelgauge.auth.together_key import TogetherApiKey
from modelgauge.general import APIException
from modelgauge.prompt import ChatPrompt, ChatRole, TextPrompt
from modelgauge.prompt_formatting import format_chat
from modelgauge.secret_values import InjectSecret
from modelgauge.sut import PromptResponseSUT, SUTOptions, SUTResponse, TokenProbability, TopTokens
from modelgauge.sut_capabilities import AcceptsChatPrompt, AcceptsTextPrompt, ProducesPerTokenLogProbabilities
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS

logger = logging.getLogger(__name__)

_SYSTEM_ROLE = "system"
_USER_ROLE = "user"
_ASSISTANT_ROLE = "assistant"

_ROLE_MAP = {
    ChatRole.user: _USER_ROLE,
    ChatRole.sut: _ASSISTANT_ROLE,
    ChatRole.system: _SYSTEM_ROLE,
}


def _retrying_request(url, headers, json_payload, method):
    """HTTP Post with retry behavior."""
    session = requests.Session()
    retries = Retry(
        total=15,
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
        allowed_methods=[method],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    if method == "POST":
        call = session.post
    elif method == "PATCH":
        call = session.patch
    elif method == "GET":
        call = session.get
    else:
        raise ValueError(f"Invalid HTTP method: {method}")
    response = None
    try:
        response = call(url, headers=headers, json=json_payload, timeout=120)
        return response
    except Exception as e:
        logger.error(f"failed on request {url} {headers} {json_payload}", exc_info=e)
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
class TogetherCompletionsSUT(PromptResponseSUT[TogetherCompletionsRequest, TogetherCompletionsResponse]):
    _URL = "https://api.together.xyz/v1/completions"

    def __init__(self, uid: str, model, api_key: TogetherApiKey):
        super().__init__(uid)
        self.model = model
        self.api_key = api_key.value

    def translate_text_prompt(self, prompt: TextPrompt, options: SUTOptions) -> TogetherCompletionsRequest:
        return self._translate_request(prompt.text, options)

    def translate_chat_prompt(self, prompt: ChatPrompt, options: SUTOptions) -> TogetherCompletionsRequest:
        return self._translate_request(format_chat(prompt, user_role=_USER_ROLE, sut_role=_ASSISTANT_ROLE), options)

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
            logprobs=options.top_logprobs,
        )

    def evaluate(self, request: TogetherCompletionsRequest) -> TogetherCompletionsResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        as_json = request.model_dump(exclude_none=True)
        response = _retrying_request(self._URL, headers, as_json, "POST")
        if not response.status_code == 200:
            raise APIException(f"Unexpected API failure ({response.status_code}): {response.text}")
        return TogetherCompletionsResponse.model_validate(response.json(), strict=True)

    def translate_response(
        self, request: TogetherCompletionsRequest, response: TogetherCompletionsResponse
    ) -> SUTResponse:
        assert len(response.choices) == 1, f"Expected 1 completion, got {len(response.choices)}."
        choice = response.choices[0]
        assert choice.text is not None
        logprobs: Optional[List[TopTokens]] = None
        if request.logprobs:
            logprobs = []
            assert choice.logprobs is not None, "Expected logprobs, but not returned."
            for token, logprob in zip(choice.logprobs.tokens, choice.logprobs.token_logprobs):
                # Together only returns 1 logprob/token
                logprobs.append(TopTokens(top_tokens=[TokenProbability(token=token, logprob=logprob)]))
        return SUTResponse(text=choice.text, top_logprobs=logprobs)


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
    _CHAT_COMPLETIONS_URL = "https://api.together.xyz/v1/chat/completions"

    def __init__(self, uid: str, model, api_key: TogetherApiKey):
        super().__init__(uid)
        self.model = model
        self.api_key = api_key.value

    def translate_text_prompt(self, prompt: TextPrompt, options: SUTOptions) -> TogetherChatRequest:
        return self._translate_request([TogetherChatRequest.Message(content=prompt.text, role=_USER_ROLE)], options)

    def translate_chat_prompt(self, prompt: ChatPrompt, options: SUTOptions) -> TogetherChatRequest:
        messages = []
        for message in prompt.messages:
            messages.append(TogetherChatRequest.Message(content=message.text, role=_ROLE_MAP[message.role]))
        return self._translate_request(messages, options)

    def _translate_request(self, messages: List[TogetherChatRequest.Message], options: SUTOptions):
        return TogetherChatRequest(
            model=self.model,
            messages=messages,
            max_tokens=options.max_tokens,
            stop=options.stop_sequences,
            temperature=options.temperature,
            top_p=options.top_p,
            top_k=options.top_k_per_token,
            repetition_penalty=options.frequency_penalty,
            logprobs=options.top_logprobs,
        )

    def evaluate(self, request: TogetherChatRequest) -> TogetherChatResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        as_json = request.model_dump(exclude_none=True)
        response = _retrying_request(self._CHAT_COMPLETIONS_URL, headers, as_json, "POST")
        if not response.status_code == 200:
            raise APIException(f"Unexpected API failure ({response.status_code}): {response.text}")
        return TogetherChatResponse.model_validate(response.json(), strict=True)

    def translate_response(self, request: TogetherChatRequest, response: TogetherChatResponse) -> SUTResponse:
        assert len(response.choices) == 1, f"Expected 1 completion, got {len(response.choices)}."
        choice = response.choices[0]
        text = choice.message.content
        assert text is not None
        logprobs: Optional[List[TopTokens]] = None
        if request.logprobs:
            logprobs = []
            assert choice.logprobs is not None, "Expected logprobs, but not returned."
            for token, logprob in zip(choice.logprobs.tokens, choice.logprobs.token_logprobs):
                # Together only returns 1 logprob/token
                logprobs.append(TopTokens(top_tokens=[TokenProbability(token=token, logprob=logprob)]))
        return SUTResponse(text=text, top_logprobs=logprobs)


@modelgauge_sut(
    capabilities=[
        AcceptsTextPrompt,
        AcceptsChatPrompt,
        ProducesPerTokenLogProbabilities,
    ]
)
class TogetherDedicatedChatSUT(TogetherChatSUT):
    """A SUT based on dedicated Together endpoint. Supports automatic endpoint spin-up."""

    _ENDPOINTS_URL = "https://api.together.xyz/v1/endpoints"

    def __init__(self, uid: str, model: str, api_key: TogetherApiKey):
        super().__init__(uid, model, api_key)
        self.endpoint_id = self._get_endpoint_id()
        self.endpoint_status = self._get_endpoint_status()

    def _get_endpoint_id(self) -> str:
        headers = {"accept": "application/json", "authorization": f"Bearer {self.api_key}"}
        response = _retrying_request(self._ENDPOINTS_URL, headers, {}, "GET")
        for endpoint in response.json()["data"]:
            if endpoint["name"] == self.model:
                return endpoint["id"]
        raise APIException(f"No endpoint found for model {self.model}")

    def _get_endpoint_status(self) -> str:
        headers = {"accept": "application/json", "authorization": f"Bearer {self.api_key}"}
        response = _retrying_request(f"{self._ENDPOINTS_URL}/{self.endpoint_id}", headers, {}, "GET")
        return response.json()["state"]

    def _spin_up_endpoint(self):
        # Get latest endpoint status
        self.endpoint_status = self._get_endpoint_status()
        if self.endpoint_status == "STARTED":
            return
        elif self.endpoint_status in ["PENDING", "STOPPING"]:
            # Wait and retry.
            time.sleep(2*60) # 2 minutes
            self._spin_up_endpoint()
        elif self.endpoint_status in ["STARTING"]:
            # Wait and retry.
            time.sleep(8*60) # 8 minutes.
            self._spin_up_endpoint()
        elif self.endpoint_status == "STOPPED" or self.endpoint_status == "ERROR":
            # Start endpoint.
            headers = {"accept": "application/json", "authorization": f"Bearer {self.api_key}"}
            payload = {"state": "STARTED"}
            response = _retrying_request(f"{self._ENDPOINTS_URL}/{self.endpoint_id}", headers, payload, "PATCH")
            self.endpoint_status = response.json()["state"]
            if self.endpoint_status != "STARTED":
                # Try again.
                self._spin_up_endpoint()

    def evaluate(self, request: TogetherChatRequest) -> TogetherChatResponse:
        if self.endpoint_status != "STARTED":
            logger.warning(f"Together endpoint for {self.model} is not ready. Status: {self.endpoint_status}. Spinning up...")
            self._spin_up_endpoint()
        try:
            return super().evaluate(request)
        except APIException as e:
            # Together returns 400 if the endpoint is not running.
            if "400" in str(e):
                logger.warning(f"Together endpoint for {self.model} is not ready. Spinning up...")
                self._spin_up_endpoint()
                return self.evaluate(request)
            else:
                raise e


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
class TogetherInferenceSUT(PromptResponseSUT[TogetherInferenceRequest, TogetherInferenceResponse]):
    _URL = "https://api.together.xyz/inference"

    def __init__(self, uid: str, model, api_key: TogetherApiKey):
        super().__init__(uid)
        self.model = model
        self.api_key = api_key.value

    def translate_text_prompt(self, prompt: TextPrompt, options: SUTOptions) -> TogetherInferenceRequest:
        return self._translate_request(prompt.text, options)

    def translate_chat_prompt(self, prompt: ChatPrompt, options: SUTOptions) -> TogetherInferenceRequest:
        return self._translate_request(format_chat(prompt, user_role=_USER_ROLE, sut_role=_ASSISTANT_ROLE), options)

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
            logprobs=options.top_logprobs,
        )

    def evaluate(self, request: TogetherInferenceRequest) -> TogetherInferenceResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        as_json = request.model_dump(exclude_none=True)
        response = _retrying_request(self._URL, headers, as_json, "POST")
        if not response.status_code == 200:
            raise APIException(f"Unexpected API failure ({response.status_code}): {response.text}")
        return TogetherInferenceResponse(**response.json())

    def translate_response(self, request: TogetherInferenceRequest, response: TogetherInferenceResponse) -> SUTResponse:
        assert len(response.output.choices) == 1, f"Expected 1 completion, got {len(response.output.choices)}."
        choice = response.output.choices[0]
        assert choice.text is not None
        logprobs: Optional[List[TopTokens]] = None
        if request.logprobs:
            logprobs = []
            assert (
                choice.tokens is not None and choice.token_logprobs is not None
            ), "Expected logprobs, but not returned."
            for token, logprob in zip(choice.tokens, choice.token_logprobs):
                logprobs.append(TopTokens(top_tokens=[TokenProbability(token=token, logprob=logprob)]))
        return SUTResponse(text=choice.text, top_logprobs=logprobs)


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
    "llama-3-70b-chat": "meta-llama/Llama-3-70b-chat-hf",
    "llama-3-70b-chat-hf": "meta-llama/Llama-3-70b-chat-hf",
    "llama-3.1-8b-instruct-turbo-together": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "llama-3.1-405b-instruct-turbo-together": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "llama-3.3-70b-instruct-turbo-together": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    "Mistral-7B-Instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
    "Mixtral-8x7B-Instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistral-8x22b-instruct": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "mistral-8x22b-instruct-nim": "mlc_ai_safety_2/nim/mistralai/mixtral-8x22b-instruct-v01-c5e6181e",
    "deepseek-R1": "deepseek-ai/DeepSeek-R1",
    "deepseek-v3-together": "deepseek-ai/DeepSeek-V3",
    # No longer supported as of 2025-04-25
    # "llama-2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    # N/A serverless on together as of 2025-03-24
    # "deepseek-llm-67b-chat": "deepseek-ai/deepseek-llm-67b-chat",
    "qwen2.5-7B-instruct-turbo-together": "Qwen/Qwen2.5-7B-Instruct-Turbo",
    # "StripedHyena-Nous-7B": "togethercomputer/StripedHyena-Nous-7B",
}
for uid, model_name in CHAT_MODELS.items():
    SUTS.register(TogetherChatSUT, uid, model_name, InjectSecret(TogetherApiKey))

DEDICATED_CHAT_MODELS = {
    "mistral-8x22b-instruct-dedicated-together": "mlc_ai_safety_2/mistralai/Mixtral-8x22B-Instruct-v0.1-26b6d754",
}
for uid, model_name in DEDICATED_CHAT_MODELS.items():
    SUTS.register(TogetherDedicatedChatSUT, uid, model_name, InjectSecret(TogetherApiKey))
