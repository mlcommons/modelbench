import logging
from typing import List, Optional

import requests  # type:ignore
from httpx import Timeout
from llama_api_client import LlamaAPIClient
from llama_api_client.types import CreateChatCompletionResponse, MessageTextContentItem, ModerationCreateResponse
from pydantic import BaseModel
from requests.adapters import HTTPAdapter, Retry  # type:ignore

from modelgauge.prompt import TextPrompt
from modelgauge.retry_decorator import retry
from modelgauge.secret_values import InjectSecret, RequiredSecret, SecretDescription
from modelgauge.sut import (
    PromptResponseSUT,
    SUTOptions,
    SUTResponse,
    REFUSAL_RESPONSE,
)
from modelgauge.sut_capabilities import AcceptsTextPrompt
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


class InputMessage(BaseModel):
    role: str
    content: str


class MetaLlamaChatRequest(BaseModel):
    model: str
    messages: List[InputMessage]
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = None


@modelgauge_sut(capabilities=[AcceptsTextPrompt])
class MetaLlamaSUT(PromptResponseSUT[MetaLlamaChatRequest, CreateChatCompletionResponse]):

    def __init__(self, uid: str, model: str, api_key: MetaLlamaApiKey):
        super().__init__(uid)
        self.model = model
        self.client = LlamaAPIClient(api_key=api_key.value, max_retries=10, timeout=Timeout(120))

    def translate_text_prompt(self, prompt: TextPrompt, options: SUTOptions) -> MetaLlamaChatRequest:
        return MetaLlamaChatRequest(
            model=self.model,
            messages=[InputMessage(role="user", content=prompt.text)],
            max_completion_tokens=options.max_tokens,
            temperature=options.temperature,
        )

    @retry()  # no obvious spurious exceptions in the code or so from some basic runs
    def evaluate(self, request: MetaLlamaChatRequest) -> CreateChatCompletionResponse:
        kwargs = request.model_dump(exclude_none=True)
        return self.client.chat.completions.create(**kwargs)

    def translate_response(self, request: MetaLlamaChatRequest, response: CreateChatCompletionResponse) -> SUTResponse:
        # type: ignore
        assert isinstance(
            response.completion_message.content, MessageTextContentItem
        ), f"unexpected response: {response}"
        text = response.completion_message.content.text
        assert text is not None
        return SUTResponse(text=text)


class MetaLlamaModeratedResponse(BaseModel):
    sut_response: CreateChatCompletionResponse
    moderation_response: ModerationCreateResponse


@modelgauge_sut(capabilities=[AcceptsTextPrompt])
class MetaLlamaModeratedSUT(PromptResponseSUT[MetaLlamaChatRequest, MetaLlamaModeratedResponse]):

    def __init__(self, uid: str, model: str, api_key: MetaLlamaApiKey):
        super().__init__(uid)
        self.model = model
        self.client = LlamaAPIClient(api_key=api_key.value, max_retries=10, timeout=Timeout(120))

    def translate_text_prompt(self, prompt: TextPrompt, options: SUTOptions) -> MetaLlamaChatRequest:
        return MetaLlamaChatRequest(
            model=self.model,
            messages=[InputMessage(role="user", content=prompt.text)],
            max_completion_tokens=options.max_tokens,
            temperature=options.temperature,
        )

    @retry()
    def evaluate(self, request: MetaLlamaChatRequest) -> MetaLlamaModeratedResponse:
        kwargs = request.model_dump(exclude_none=True)
        chat_response = self.client.chat.completions.create(**kwargs)
        messages: list = kwargs.get("messages")  # type: ignore
        messages.append(chat_response.completion_message)
        moderation_response = self.client.moderations.create(messages=messages)
        for r in moderation_response.results:
            if r.flagged_categories is None:
                # make objects comply with Pydantic definitions due to bug;
                # see https://github.com/meta-llama/llama-api-python/issues/33 for more
                r.flagged_categories = []
        return MetaLlamaModeratedResponse(sut_response=chat_response, moderation_response=moderation_response)

    def translate_response(self, request: MetaLlamaChatRequest, response: MetaLlamaModeratedResponse) -> SUTResponse:
        for m in response.moderation_response.results:
            if m.flagged:
                return SUTResponse(text=REFUSAL_RESPONSE)
        # type: ignore
        assert isinstance(
            response.sut_response.completion_message.content, MessageTextContentItem
        ), f"unexpected response: {response}"
        text = response.sut_response.completion_message.content.text
        assert text is not None
        return SUTResponse(text=text)


CHAT_MODELS = ["Llama-4-Scout-17B-16E-Instruct-FP8", "Llama-4-Maverick-17B-128E-Instruct-FP8", "Llama-3.3-8B-Instruct"]

for model_name in CHAT_MODELS:
    SUTS.register(MetaLlamaSUT, "meta-" + model_name.lower() + "-llama", model_name, InjectSecret(MetaLlamaApiKey))
    SUTS.register(
        MetaLlamaModeratedSUT,
        "meta-" + model_name.lower() + "-moderated-llama",
        model_name,
        InjectSecret(MetaLlamaApiKey),
    )
