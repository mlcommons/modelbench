from abc import ABC, abstractmethod
from typing import List, Optional

import requests  # type:ignore
from pydantic import BaseModel
from requests.adapters import HTTPAdapter, Retry  # type:ignore

from modelgauge.general import APIException
from modelgauge.prompt import TextPrompt
from modelgauge.secret_values import InjectSecret, RequiredSecret, SecretDescription
from modelgauge.sut import PromptResponseSUT, SUTOptions, SUTResponse
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS


# TODO: Unify with Together client retry logic.
def _retrying_post(url, headers, json_payload):
    """HTTP Post with retry behavior."""

    session = requests.Session()
    retries = Retry(
        total=7,
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


class AzureApiKey(RequiredSecret, ABC):
    # Different endpoints may have different api keys.

    @classmethod
    @abstractmethod
    def scope(cls) -> str:
        """The scope name for a specific azure endpoint API key."""
        pass

    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope=cls.scope(),
            key="api_key",
            instructions="Ask MLCommons admin for permission.",
        )


class AzureChatRequest(BaseModel):
    class Message(BaseModel):
        role: str
        content: str

    messages: List[Message]
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None


class AzureChatResponse(BaseModel):
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


@modelgauge_sut(capabilities=[AcceptsTextPrompt])
class AzureChatSUT(PromptResponseSUT[AzureChatRequest, AzureChatResponse]):
    def __init__(self, uid: str, endpoint_url: str, api_key: AzureApiKey):
        # TODO: Secret must be generalized.
        super().__init__(uid)
        self.endpoint_url = endpoint_url
        self.api_key = api_key.value

    def translate_text_prompt(self, prompt: TextPrompt, options: SUTOptions) -> AzureChatRequest:
        messages = [AzureChatRequest.Message(content=prompt.text, role="user")]
        return AzureChatRequest(
            messages=messages,
            max_tokens=options.max_tokens,
            stop=options.stop_sequences,
            temperature=options.temperature,
            top_p=options.top_p,
            frequency_penalty=options.frequency_penalty,
            presence_penalty=options.presence_penalty,
        )

    def evaluate(self, request: AzureChatRequest) -> AzureChatResponse:
        headers = {"Authorization": self.api_key}
        as_json = request.model_dump(exclude_none=True)
        response = _retrying_post(f"{self.endpoint_url}/v1/chat/completions", headers, as_json)
        if not response.status_code == 200:
            raise APIException(f"Unexpected API failure from SUT {self.uid} ({response.status_code}): {response.text}")
        return AzureChatResponse.model_validate(response.json(), strict=True)

    def translate_response(self, request: AzureChatRequest, response: AzureChatResponse) -> SUTResponse:
        assert len(response.choices) == 1, f"Expected a single response message, got {len(response.choices)}."
        text = response.choices[0].message.content
        assert text is not None
        return SUTResponse(text=text)


class PhiMiniKey(AzureApiKey):
    @classmethod
    def scope(cls) -> str:
        return "azure_phi_3_5_mini_endpoint"


class PhiMoEKey(AzureApiKey):
    @classmethod
    def scope(cls) -> str:
        return "azure_phi_3_5_moe_endpoint"


class Phi4Key(AzureApiKey):
    @classmethod
    def scope(cls) -> str:
        return "azure_phi_4_endpoint"


SUTS.register(
    AzureChatSUT,
    "phi-3.5-mini-instruct",
    "https://Phi-3-5-mini-instruct-hfkpb.eastus2.models.ai.azure.com",
    InjectSecret(PhiMiniKey),
)

SUTS.register(
    AzureChatSUT,
    "phi-3.5-moe-instruct",
    "https://Phi-3-5-MoE-instruct-abneb.eastus2.models.ai.azure.com",
    InjectSecret(PhiMoEKey),
)

SUTS.register(
    AzureChatSUT,
    "phi-4",
    "https://Phi-4-gckxq.eastus2.models.ai.azure.com",
    InjectSecret(Phi4Key),
)

SUTS.register(
    AzureChatSUT,
    "ai21labs-AI21-Jamba-Large-1.5-azure",
    "https://AI21-Jamba-1-5-Large-wwkju.eastus2.models.ai.azure.com/chat/completions",
    InjectSecret(Phi4Key),
)
