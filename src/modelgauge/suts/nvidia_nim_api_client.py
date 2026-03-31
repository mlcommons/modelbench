from typing import Optional

from modelgauge.secret_values import (
    InjectSecret,
    RequiredSecret,
    SecretDescription,
)
from modelgauge.suts.openai_client import OpenAIChat, OpenAIChatRequest
from modelgauge.model_options import ModelOptions
from modelgauge.sut_capabilities import (
    AcceptsChatPrompt,
    AcceptsTextPrompt,
)
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS

BASE_URL = "https://integrate.api.nvidia.com/v1"


class NvidiaNIMApiKey(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="nvidia-nim-api",
            key="api_key",
            instructions="See https://build.nvidia.com/",
        )


class NIMOpenAIChatRequest(OpenAIChatRequest):
    max_tokens: Optional[int] = (
        256  # NVIDIA NIM uses the deprecated "max_tokens" param name instead of "max_completion_tokens"
    )


@modelgauge_sut(
    capabilities=[
        AcceptsTextPrompt,
        AcceptsChatPrompt,
    ]
)
class NvidiaNIMApiClient(OpenAIChat):
    """
    Documented at https://https://docs.api.nvidia.com/
    """

    def __init__(self, uid: str, model: str, api_key: NvidiaNIMApiKey):
        super().__init__(uid, model, api_key=api_key, base_url=BASE_URL)

    def _translate_request(self, messages, options: ModelOptions) -> NIMOpenAIChatRequest:
        request = super()._translate_request(messages, options)
        request_json = request.model_dump(exclude_none=True)
        del request_json["max_completion_tokens"]  # NIM API doesn't allow extra inputs
        return NIMOpenAIChatRequest(
            max_tokens=options.max_tokens,
            **request_json,
        )


SUTS.register(
    NvidiaNIMApiClient,
    "nvidia-nemotron-mini-4b-instruct",
    "nvidia/nemotron-mini-4b-instruct",
    InjectSecret(NvidiaNIMApiKey),
)
