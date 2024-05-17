from typing import Any, List, Dict

import together
from modelgauge.prompt import ChatPrompt, TextPrompt
from modelgauge.secret_values import RequiredSecret, SecretDescription
from modelgauge.sut import PromptResponseSUT, RequestType, ResponseType, SUTResponse, SUTCompletion
from modelgauge.sut_capabilities import AcceptsTextPrompt, AcceptsChatPrompt
from modelgauge.sut_decorator import modelgauge_sut
from pydantic import BaseModel
from together.abstract import api_requestor
from together.types import TogetherRequest, ChatCompletionResponse


class TogetherApiKey(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="together",
            key="api_key",
            instructions="See https://api.together.xyz/settings/api-keys",
        )


class TogetherChatRequest(BaseModel):
    messages: List[Dict]


class TogetherChatResponse(BaseModel):
    text: str


@modelgauge_sut(capabilities=[AcceptsTextPrompt])
class TogetherChatSUT(PromptResponseSUT[TogetherChatRequest, TogetherChatResponse]):
    def __init__(self, uid: str, model, api_key: TogetherApiKey):
        super().__init__(uid)
        self.model = model
        client = together.Together(api_key=api_key.value)
        self.requestor = api_requestor.APIRequestor(client=client.client)

    def translate_text_prompt(self, prompt: TextPrompt) -> RequestType:
        return TogetherChatRequest(messages=[{"role": "user", "content": prompt.text}])

    def evaluate(self, request: TogetherChatRequest) -> TogetherChatResponse:
        params = dict(model=self.model, messages=request.messages)
        request = TogetherRequest(method="POST", url="chat/completions", params=params)
        raw_result = self.requestor.request(options=request)[0]
        return TogetherChatResponse(text=ChatCompletionResponse(**raw_result.data).choices[0].message.content)

    def translate_response(self, request: RequestType, response: ResponseType) -> SUTResponse:
        return SUTResponse(completions=[SUTCompletion(text=response.text)])
