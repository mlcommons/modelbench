from typing import List, Optional

import requests  # type: ignore

from modelgauge.prompt import TextPrompt
from modelgauge.retry_decorator import retry
from modelgauge.secret_values import InjectSecret, RequiredSecret, SecretDescription
from modelgauge.sut import PromptResponseSUT, SUTOptions, SUTResponse
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS
from pydantic import BaseModel


class BasetenChatMessage(BaseModel):
    content: str
    role: str


class BasetenChatRequest(BaseModel):
    model: str
    stream: Optional[bool] = False


class BasetenChatMessagesRequest(BasetenChatRequest):
    messages: List[BasetenChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    frequency_penalty: Optional[int] = None


class BasetenChatPromptRequest(BasetenChatRequest):
    prompt: str
    max_tokens: Optional[int] = None


class BasetenResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[dict]
    usage: dict


class BasetenInferenceAPIKey(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="baseten",
            key="api_key",
            instructions="You can create an api key at https://app.baseten.co/settings/api_keys .",
        )


class BasetenSUT(PromptResponseSUT[BasetenChatRequest, BasetenResponse]):
    """A SUT that is hosted on a dedicated Baseten inference endpoint."""

    HOST = "baseten"

    def __init__(self, uid: str, model: str, endpoint: str, key: BasetenInferenceAPIKey):
        super().__init__(uid)
        self.key = key.value
        self.model = model
        self.endpoint = endpoint

    @retry(transient_exceptions=[requests.exceptions.ConnectionError])
    def evaluate(self, request: BasetenChatRequest) -> BasetenResponse:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Api-Key {self.key}",
            "Content-Type": "application/json",
        }
        data = request.model_dump(exclude_none=True)
        response = requests.post(self.endpoint, headers=headers, json=data)
        try:
            if response.status_code != 200:
                response.raise_for_status()
            response_data = response.json()
            eval_response = (
                BasetenResponse(**response_data)
                if type(response_data) == dict
                else BasetenResponse(text=str(response_data))
            )
            return eval_response
        except Exception as e:
            print(f"Unexpected failure for {data}: {response}:\n {response.content}\n{response.headers}")
            raise e

    def translate_response(self, request: BasetenChatRequest, response: BasetenResponse) -> SUTResponse:
        assert len(response.choices) == 1, f"Expected exactly one choice, got {len(response.choices)}."
        return SUTResponse(text=response.choices[0]["message"]["content"])


@modelgauge_sut(capabilities=[AcceptsTextPrompt])
class BasetenPromptSUT(BasetenSUT):
    def translate_text_prompt(self, prompt: TextPrompt, options: SUTOptions) -> BasetenChatRequest:
        return BasetenChatPromptRequest(
            model=self.model, prompt=prompt.text, stream=False, max_tokens=options.max_tokens
        )


@modelgauge_sut(capabilities=[AcceptsTextPrompt])
class BasetenMessagesSUT(BasetenSUT):
    def translate_text_prompt(self, prompt: TextPrompt, options: SUTOptions) -> BasetenChatRequest:
        return BasetenChatMessagesRequest(
            model=self.model,
            messages=[BasetenChatMessage(role="user", content=prompt.text)],
            stream=False,
            max_tokens=options.max_tokens,
            temperature=options.temperature,
            top_p=options.top_p,
            top_k=options.top_k_per_token,
            frequency_penalty=options.frequency_penalty,
        )


BASETEN_SECRET = InjectSecret(BasetenInferenceAPIKey)

SUTS.register(
    BasetenMessagesSUT,
    "nvidia-llama-3.3-49b-nemotron-super",
    "nvidia/llama-3.3-nemotron-super-49b-v1",
    "https://model-v319m4rq.api.baseten.co/environments/production/predict",
    BASETEN_SECRET,
)
