import time
from dataclasses import dataclass
from typing import List, Optional

import requests  # type: ignore
from airrlogger.log_config import get_logger
from pydantic import BaseModel
from requests.adapters import HTTPAdapter, Retry  # type: ignore

from modelgauge.auth.together_secrets import TogetherApiKey, TogetherProjectId
from modelgauge.general import APIException
from modelgauge.model_options import ModelOptions, TokenProbability, TopTokens
from modelgauge.prompt import ChatPrompt, ChatRole, TextPrompt
from modelgauge.prompt_formatting import format_chat
from modelgauge.reasoning_handlers import ThinkingMixin
from modelgauge.secret_values import InjectSecret
from modelgauge.sut import PromptResponseSUT, SUTResponse
from modelgauge.sut_capabilities import AcceptsChatPrompt, AcceptsTextPrompt, ProducesPerTokenLogProbabilities
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS

logger = get_logger(__name__)

_SYSTEM_ROLE = "system"
_USER_ROLE = "user"
_ASSISTANT_ROLE = "assistant"

_ROLE_MAP = {
    ChatRole.user: _USER_ROLE,
    ChatRole.sut: _ASSISTANT_ROLE,
    ChatRole.system: _SYSTEM_ROLE,
}


@dataclass
class TogetherEndpointState:
    STARTED: str = "DEPLOYMENT_STATE_READY"
    STOPPED: str = "DEPLOYMENT_STATE_STOPPED"
    ERROR: str = "DEPLOYMENT_STATE_FAILED"
    DEGRADED: str = "DEPLOYMENT_STATE_DEGRADED"
    PENDING: str = "DEPLOYMENT_STATE_PROVISIONING"
    STARTING: str = "DEPLOYMENT_STATE_SCALING"
    STOPPING: str = "DEPLOYMENT_STATE_STOPPING"


def _retrying_request(url, headers, json_payload, method, params=None):
    """HTTP Post with retry behavior."""
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
    with requests.Session() as session:
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
            kwargs = {"headers": headers, "timeout": 120}
            if json_payload:
                kwargs["json"] = json_payload
            if params:
                kwargs["params"] = params
            response = call(url, **kwargs)
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
    max_tokens: int = 100
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
class TogetherCompletionsSUT(PromptResponseSUT):
    _URL = "https://api.together.xyz/v1/completions"

    def __init__(self, uid: str, model, api_key: TogetherApiKey):
        super().__init__(uid)
        self.model = model
        self.api_key = api_key.value

    def translate_text_prompt(self, prompt: TextPrompt, options: ModelOptions) -> TogetherCompletionsRequest:
        return self._translate_request(prompt.text, options)

    def translate_chat_prompt(self, prompt: ChatPrompt, options: ModelOptions) -> TogetherCompletionsRequest:
        return self._translate_request(format_chat(prompt, user_role=_USER_ROLE, sut_role=_ASSISTANT_ROLE), options)

    def _translate_request(self, text, options):
        exclude_none_kwargs = {}
        if options.max_tokens is not None:
            exclude_none_kwargs["max_tokens"] = options.max_tokens
        return TogetherCompletionsRequest(
            model=self.model,
            prompt=text,
            stop=options.stop_sequences,
            temperature=options.temperature,
            top_p=options.top_p,
            top_k=options.top_k_per_token,
            repetition_penalty=options.frequency_penalty,
            logprobs=options.top_logprobs,
            **exclude_none_kwargs,
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
class TogetherChatSUT(PromptResponseSUT):
    _CHAT_COMPLETIONS_URL = "https://api.together.xyz/v1/chat/completions"

    def __init__(self, uid: str, model, api_key: TogetherApiKey):
        super().__init__(uid)
        self.model = model
        self.api_key = api_key.value

    def translate_text_prompt(self, prompt: TextPrompt, options: ModelOptions) -> TogetherChatRequest:
        return self._translate_request([TogetherChatRequest.Message(content=prompt.text, role=_USER_ROLE)], options)

    def translate_chat_prompt(self, prompt: ChatPrompt, options: ModelOptions) -> TogetherChatRequest:
        messages = []
        for message in prompt.messages:
            messages.append(TogetherChatRequest.Message(content=message.text, role=_ROLE_MAP[message.role]))
        return self._translate_request(messages, options)

    def _translate_request(self, messages: List[TogetherChatRequest.Message], options: ModelOptions):
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


@modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
class TogetherThinkingSUT(ThinkingMixin, TogetherChatSUT):
    """SUT that preforms reasoning like deepseek-r1"""


@modelgauge_sut(
    capabilities=[
        AcceptsTextPrompt,
        AcceptsChatPrompt,
        ProducesPerTokenLogProbabilities,
    ]
)
class TogetherDedicatedChatSUT(TogetherChatSUT):
    """A SUT based on dedicated Together endpoint. Supports automatic endpoint spin-up."""

    def __init__(self, uid: str, model: str, api_key: TogetherApiKey, project_id: TogetherProjectId):
        super().__init__(uid, model, api_key)
        self.project_id = project_id.value
        self.endpoint_id: Optional[str] = None
        self.deployment_id: Optional[str] = None
        self.endpoint_status: Optional[str] = None

        # Can't lazy init because we need to know the endpoint info (the model name, speicifically) to translate the request.
        self._set_endpoint_info()

    def _endpoints_url(self) -> str:
        return f"https://api.together.ai/v2/projects/{self.project_id}/endpoints"

    def _set_endpoint_info(self):
        headers = {"accept": "application/json", "authorization": f"Bearer {self.api_key}"}
        response = _retrying_request(self._endpoints_url(), headers, None, "GET")
        for endpoint in response.json()["data"]:
            for deployment in endpoint["deployments"]:
                if self.model.lower() in deployment["name"].lower():
                    self.endpoint_id = endpoint["id"]
                    self.deployment_id = deployment["id"]
                    self.model = endpoint["name"]
                    return
        raise APIException(f"No endpoint found for model {self.model}")

    def _get_endpoint_status(self) -> str:
        headers = {"accept": "application/json", "authorization": f"Bearer {self.api_key}"}
        response = _retrying_request(
            f"{self._endpoints_url()}/{self.endpoint_id}/deployments/{self.deployment_id}", headers, {}, "GET"
        )
        return response.json()["status"]["state"]

    def _spin_up_endpoint(self):
        # TODO: Add a warning about manually stopping endpoiint somewhere.
        # Get latest endpoint status
        self.endpoint_status = self._get_endpoint_status()
        if self.endpoint_status == TogetherEndpointState.STARTED:
            return
        if self.endpoint_status == TogetherEndpointState.DEGRADED:
            raise APIException(f"Together endpoint for {self.model} is degraded. Status: {self.endpoint_status}.")
        elif self.endpoint_status in [
            TogetherEndpointState.PENDING,
            TogetherEndpointState.STOPPING,
            TogetherEndpointState.STARTING,
        ]:
            # Wait and retry.
            time.sleep(2 * 60)  # 2 minutes
            self._spin_up_endpoint()
        elif self.endpoint_status in [TogetherEndpointState.STOPPED, TogetherEndpointState.ERROR]:
            # Start endpoint.
            logger.warning(
                f"Together endpoint for {self.model} is not ready. Status: {self.endpoint_status}. Spinning up..."
            )
            headers = {"accept": "application/json", "authorization": f"Bearer {self.api_key}"}
            params = {"update_mask": "autoscaling.minReplicas,autoscaling.maxReplicas"}
            payload = {"autoscaling": {"minReplicas": 1, "maxReplicas": 1}}
            response = _retrying_request(
                f"{self._endpoints_url()}/{self.endpoint_id}/deployments/{self.deployment_id}",
                headers,
                payload,
                "PATCH",
                params=params,
            )
            state = response.json().get("status", {}).get("state", None)
            if state is not None:
                self.endpoint_status = state
            if self.endpoint_status != TogetherEndpointState.STARTED:
                # Try again.
                self._spin_up_endpoint()

    def evaluate(self, request: TogetherChatRequest) -> TogetherChatResponse:
        if self.endpoint_status != TogetherEndpointState.STARTED:
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

DEDICATED_CHAT_MODELS = {}
for uid, model_name in DEDICATED_CHAT_MODELS.items():
    SUTS.register(
        TogetherDedicatedChatSUT, uid, model_name, InjectSecret(TogetherApiKey), InjectSecret(TogetherProjectId)
    )

SUTS.register(TogetherThinkingSUT, "deepseek-R1-thinking", "deepseek-ai/DeepSeek-R1", InjectSecret(TogetherApiKey))
