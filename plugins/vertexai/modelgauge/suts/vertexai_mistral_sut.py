from typing import Dict, Optional

from modelgauge.prompt import TextPrompt
from modelgauge.secret_values import InjectSecret
from modelgauge.sut import PromptResponseSUT, SUTOptions, SUTResponse
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS
from modelgauge.suts.vertexai_client import (
    VertexAIClient,
    VertexAIProjectId,
    VertexAIRegion,
)
from pydantic import BaseModel, ConfigDict

_USER_ROLE = "user"


class VertexAIMistralRequest(BaseModel):
    # https://docs.mistral.ai/deployment/cloud/vertex/
    model: str
    messages: list[dict]
    # TODO: to guard against defaults changing, we may want to make
    # the following fields required, so the user/client is forced to
    # affirmatively specify them (for reproducibility)
    temperature: Optional[float] = None
    response_format: Optional[Dict[str, str]] = None
    safe_prompt: Optional[bool] = True
    stream: Optional[bool] = False
    max_tokens: Optional[int]


class VertexAIMistralResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    object: str
    model: str
    created: int
    choices: list[Dict]
    usage: Dict[str, int]


@modelgauge_sut(capabilities=[AcceptsTextPrompt])
class VertexAIMistralAISut(PromptResponseSUT):
    """A MistralAI SUT hosted on GCP's Vertex service."""

    def __init__(
        self,
        uid: str,
        model_name: str,
        model_version: str,
        project_id: VertexAIProjectId,
        region: VertexAIRegion,
    ):
        super().__init__(uid)
        self.model_name = model_name
        self.model_version = model_version
        self._project_id = project_id
        self._region = region
        self._client = None

    @property
    def client(self) -> VertexAIClient:
        if not self._client:
            self._client = VertexAIClient(
                publisher="mistralai",
                model_name=self.model_name,
                model_version=self.model_version,
                streaming=False,
                project_id=self._project_id,
                region=self._region,
            )
        return self._client

    def translate_text_prompt(self, prompt: TextPrompt, options: SUTOptions) -> VertexAIMistralRequest:
        args = {"model": self.model_name, "messages": [{"role": _USER_ROLE, "content": prompt.text}]}
        if options.temperature is not None:
            args["temperature"] = options.temperature
        if options.max_tokens is not None:
            args["max_tokens"] = options.max_tokens
        return VertexAIMistralRequest(**args)

    def evaluate(self, request: VertexAIMistralRequest) -> VertexAIMistralResponse:
        response = self.client.request(request.model_dump(exclude_none=True))  # type: ignore
        return VertexAIMistralResponse(**response)

    def translate_response(self, request: VertexAIMistralRequest, response: VertexAIMistralResponse) -> SUTResponse:
        assert len(response.choices) == 1, f"Expected 1 completion, got {len(response.choices)}."
        completions = []
        text = response.choices[0]["message"]["content"]
        assert text is not None
        return SUTResponse(text=text)


VERTEX_PROJECT_ID = InjectSecret(VertexAIProjectId)
VERTEX_REGION = InjectSecret(VertexAIRegion)

model_name = "mistral-large"
model_version = "2407"
model_uid = f"vertexai-{model_name}-{model_version}"
# If you prefer to use MistralAI, please see plugins/mistral
# Authentication required using https://cloud.google.com/docs/authentication/application-default-credentials
SUTS.register(VertexAIMistralAISut, model_uid, model_name, model_version, VERTEX_PROJECT_ID, VERTEX_REGION)
