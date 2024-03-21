from typing import Any, List, Optional
from pydantic import BaseModel, Field
import requests
from requests.adapters import HTTPAdapter, Retry
from together.utils import response_status_exception  # type: ignore
from newhelm.prompt import ChatPrompt, ChatRole, SUTOptions, TextPrompt
from newhelm.prompt_formatting import format_chat
from newhelm.secret_values import (
    InjectSecret,
    RequiredSecret,
    SecretDescription,
)
from newhelm.sut import PromptResponseSUT, SUTCompletion, SUTResponse
from newhelm.sut_decorator import newhelm_sut

from newhelm.sut_registry import SUTS


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
    response = session.post(url, headers=headers, json=json_payload)
    try:
        response_status_exception(response)
    except Exception as e:
        raise Exception(
            f"Exception calling {url} with {json_payload}. Response {response.text}"
        ) from e
    return response


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


class TogetherCompletionsResponse(BaseModel):
    # https://docs.together.ai/reference/completions
    class Choice(BaseModel):
        text: str

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


@newhelm_sut()
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
        )

    def evaluate(
        self, request: TogetherCompletionsRequest
    ) -> TogetherCompletionsResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        as_json = request.model_dump(exclude_none=True)
        response = _retrying_post(self._URL, headers, as_json)
        return TogetherCompletionsResponse.model_validate(response.json(), strict=True)

    def translate_response(
        self, request: TogetherCompletionsRequest, response: TogetherCompletionsResponse
    ) -> SUTResponse:
        sut_completions = []
        for choice in response.choices:
            assert choice.text is not None
            sut_completions.append(SUTCompletion(text=choice.text))
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


class TogetherChatResponse(BaseModel):
    # https://docs.together.ai/reference/chat-completions
    class Choice(BaseModel):
        class Message(BaseModel):
            role: str
            content: str

        message: Message

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


@newhelm_sut()
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
        )

    def evaluate(self, request: TogetherChatRequest) -> TogetherChatResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        as_json = request.model_dump(exclude_none=True)
        response = _retrying_post(self._URL, headers, as_json)
        response_status_exception(response)
        return TogetherChatResponse.model_validate(response.json(), strict=True)

    def translate_response(
        self, request: TogetherChatRequest, response: TogetherChatResponse
    ) -> SUTResponse:
        sut_completions = []
        for choice in response.choices:
            text = choice.message.content
            assert text is not None
            sut_completions.append(SUTCompletion(text=text))
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


class TogetherInferenceResponse(BaseModel):
    class Args(BaseModel):
        model: str
        prompt: Optional[str] = None
        temperature: float
        top_p: float
        top_k: float
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

        choices: List[Choice]
        raw_compute_time: Optional[float] = None
        result_type: str

    output: Output


@newhelm_sut()
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
        )

    def evaluate(self, request: TogetherInferenceRequest) -> TogetherInferenceResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        as_json = request.model_dump(exclude_none=True)
        response = _retrying_post(self._URL, headers, as_json)
        response_status_exception(response)
        return TogetherInferenceResponse(**response.json())

    def translate_response(
        self, request: TogetherInferenceRequest, response: TogetherInferenceResponse
    ) -> SUTResponse:
        for p in response.prompt:
            print(p)
        sut_completions = []
        for choice in response.output.choices:
            assert choice.text is not None
            sut_completions.append(SUTCompletion(text=choice.text))
        return SUTResponse(completions=sut_completions)


API_KEY_SECRET = InjectSecret(TogetherApiKey)

# Language
SUTS.register(
    TogetherCompletionsSUT, "llama-2-7b", "togethercomputer/llama-2-7b", API_KEY_SECRET
)
SUTS.register(
    TogetherCompletionsSUT,
    "llama-2-70b",
    "togethercomputer/llama-2-70b",
    API_KEY_SECRET,
)
SUTS.register(
    TogetherCompletionsSUT, "falcon-40b", "togethercomputer/falcon-40b", API_KEY_SECRET
)
SUTS.register(
    TogetherCompletionsSUT,
    "llama-2-13b",
    "togethercomputer/llama-2-13b",
    API_KEY_SECRET,
)
SUTS.register(TogetherCompletionsSUT, "flan-t5-xl", "google/flan-t5-xl", API_KEY_SECRET)


# Chat
SUTS.register(
    TogetherChatSUT,
    "llama-2-7b-chat",
    "togethercomputer/llama-2-7b-chat",
    API_KEY_SECRET,
)
SUTS.register(
    TogetherChatSUT,
    "llama-2-70b-chat",
    "togethercomputer/llama-2-70b-chat",
    API_KEY_SECRET,
)
SUTS.register(
    TogetherChatSUT, "zephyr-7b-beta", "HuggingFaceH4/zephyr-7b-beta", API_KEY_SECRET
)
SUTS.register(
    TogetherChatSUT, "vicuna-13b-v1.5", "lmsys/vicuna-13b-v1.5", API_KEY_SECRET
)
SUTS.register(
    TogetherChatSUT,
    "Mistral-7B-Instruct-v0.2",
    "mistralai/Mistral-7B-Instruct-v0.2",
    API_KEY_SECRET,
)
SUTS.register(
    TogetherChatSUT, "WizardLM-13B-V1.2", "WizardLM/WizardLM-13B-V1.2", API_KEY_SECRET
)
SUTS.register(
    TogetherChatSUT,
    "oasst-sft-4-pythia-12b-epoch-3.5",
    "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
    API_KEY_SECRET,
)
SUTS.register(
    TogetherChatSUT, "dolly-v2-12b", "databricks/dolly-v2-12b", API_KEY_SECRET
)
