import logging
from typing import List, Optional

import requests  # type:ignore
from llama_api_client import LlamaAPIClient
from llama_api_client.types import CreateChatCompletionResponse
from pydantic import BaseModel
from requests.adapters import HTTPAdapter, Retry  # type:ignore

from modelgauge.prompt import TextPrompt
from modelgauge.secret_values import InjectSecret, RequiredSecret, SecretDescription
from modelgauge.sut import (
    PromptResponseSUT,
    SUTOptions,
    SUTResponse,
    RequestType,
)
from modelgauge.sut_capabilities import AcceptsChatPrompt, AcceptsTextPrompt
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS

logger = logging.getLogger(__name__)


class MetaLlamaApiKey(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="meta_llama",
            key="api_key",
            instructions="See https://llama.developer.meta.com/docs/api-keys/",
        )


class ContentItem(BaseModel):
    type: str = "text"
    text: str


class InputMessage(BaseModel):
    role: str
    content: str


class OutputMessage(BaseModel):
    role: str
    stop_reason: str
    content: ContentItem


class MetaLlamaChatRequest(BaseModel):
    model: str
    messages: List[InputMessage]
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = None


@modelgauge_sut(
    capabilities=[
        AcceptsTextPrompt,
    ]
)
class MetaLlamaSUT(PromptResponseSUT[MetaLlamaChatRequest, CreateChatCompletionResponse]):

    def __init__(self, uid: str, model: str, api_key: MetaLlamaApiKey):
        super().__init__(uid)
        self.model = model
        self.client = LlamaAPIClient(api_key=api_key.value)

    def translate_text_prompt(self, prompt: TextPrompt, options: SUTOptions) -> RequestType:
        return MetaLlamaChatRequest(
            model=self.model,
            messages=[InputMessage(role="user", content=prompt.text)],
            max_completion_tokens=options.max_tokens,
            temperature=options.temperature,
        )

    def evaluate(self, request: MetaLlamaChatRequest) -> CreateChatCompletionResponse:
        kwargs = dict(request)
        if request.temperature is None:
            # this violates the JSON schema on the server, and they use their own JSON transformer that we can't override
            del kwargs["temperature"]
        return self.client.chat.completions.create(**kwargs)

    def translate_response(self, request: MetaLlamaChatRequest, response: CreateChatCompletionResponse) -> SUTResponse:
        text = response.completion_message.content.text
        assert text is not None
        return SUTResponse(text=response.completion_message.content.text)


CHAT_MODELS = ["Llama-4-Scout-17B-16E-Instruct-FP8", "Llama-4-Maverick-17B-128E-Instruct-FP8", "Llama-3.3-8B-Instruct"]

for model_name in CHAT_MODELS:
    uid = "meta-" + model_name.lower() + "-llama"
    SUTS.register(MetaLlamaSUT, uid, model_name, InjectSecret(MetaLlamaApiKey))
