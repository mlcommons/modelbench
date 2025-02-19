from typing import Optional, List
import os

import requests  # type: ignore
from pydantic import BaseModel

from modelgauge.prompt import TextPrompt
from modelgauge.secret_values import InjectSecret, RequiredSecret, SecretDescription
from modelgauge.sut import PromptResponseSUT, SUTCompletion, SUTResponse
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS


class BasetenChatMessage(BaseModel):
    content: str
    role: str

class BasetenChatRequest(BaseModel):
    stream: Optional[bool] = False

class BasetenChatMessagesRequest(BasetenChatRequest):
    messages: List[BasetenChatMessage]
    max_new_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    frequency_penalty: Optional[int] = None

class BasetenChatPromptRequest(BasetenChatRequest):
    prompt: str
    max_tokens: Optional[int] = 2048

class BasetenResponse(BaseModel):
    text: str

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

    def __init__(self, uid: str, endpoint: str, key: BasetenInferenceAPIKey):
        super().__init__(uid)
        self.key = key.value
        self.endpoint = endpoint

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
            eval_response = BasetenResponse(**response_data) if type(response_data)==dict else BasetenResponse(text=str(response_data))
            return eval_response
        except Exception as e:
            print(f"Unexpected failure for {data}: {response}:\n {response.content}\n{response.headers}")
            raise e

    def translate_response(self, request: BasetenChatRequest, response: BasetenResponse) -> SUTResponse:
        return SUTResponse(completions=[SUTCompletion(text=response.text)])


@modelgauge_sut(capabilities=[AcceptsTextPrompt])
class BasetenPromptSUT(BasetenSUT):
    def __init__(self, uid: str, endpoint: str, key: BasetenInferenceAPIKey):
        super().__init__(uid,endpoint,key)

    def translate_text_prompt(self, prompt: TextPrompt) -> BasetenChatRequest:
        return BasetenChatPromptRequest(
            prompt=prompt.text,
            stream=False,
            max_tokens=prompt.options.max_tokens
        )

@modelgauge_sut(capabilities=[AcceptsTextPrompt])
class BasetenMessagesSUT(BasetenSUT):
    def __init__(self, uid: str, endpoint: str, key: BasetenInferenceAPIKey):
        super().__init__(uid,endpoint,key)

    def translate_text_prompt(self, prompt: TextPrompt) -> BasetenChatRequest:
        return BasetenChatMessagesRequest(
            messages=[BasetenChatMessage(role='user',content=prompt.text)],
            stream=False,
            max_new_tokens=prompt.options.max_tokens, 
            temperature=prompt.options.temperature,
            top_p=prompt.options.top_p,
            top_k=prompt.options.top_k_per_token,
            frequency_penalty=prompt.options.frequency_penalty
        )

BASETEN_SECRET = InjectSecret(BasetenInferenceAPIKey)

for item in os.environ.get('BASETEN_MODELS','').split(','):
    item = item.strip()
    if len(item)==0:
        continue
    uid, _, model = item.partition('=')

    model, _, kind = model.partition(';')
    match kind.strip():
        case 'prompt':
            sut_class = BasetenPromptSUT
        case 'messages':
            sut_class = BasetenMessagesSUT
        case _:
            sut_class = BasetenMessagesSUT


    endpoint = f'https://model-{model.strip()}.api.baseten.co/production/predict'

    SUTS.register(
        sut_class,
        uid,
        endpoint,
        BASETEN_SECRET,
    )
