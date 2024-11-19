from typing import Dict, Optional
from xml.parsers.expat import model

from modelgauge.prompt import TextPrompt
from modelgauge.secret_values import InjectSecret
from modelgauge.sut import PromptResponseSUT, SUTCompletion, SUTResponse
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS
from modelgauge.suts.vertexai_client import (
    VertexAIAPIKey,
    VertexAIClient,
    VertexAIProjectId,
)
from pydantic import BaseModel, ConfigDict

_USER_ROLE = "user"


class MistralRequest(BaseModel):
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
    n: int = 1


class MistralResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    object: str
    model: str
    created: int
    choices: list[Dict]
    usage: Dict[str, int]


@modelgauge_sut(capabilities=[AcceptsTextPrompt])
class MistralAISut(PromptResponseSUT):
    """A MistralAI SUT hosted on GCP's Vertex service."""

    def __init__(
        self, uid: str, model_name: str, model_version: str, api_key: VertexAIAPIKey, project_id: VertexAIProjectId
    ):
        super().__init__(uid)
        self.model_name = model_name
        self.model_version = model_version
        self._api_key = api_key
        self._project_id = project_id
        self._client = None

    @property
    def client(self) -> VertexAIClient:
        if not self._client:
            self._client = VertexAIClient(
                publisher="mistralai",
                model_name=self.model_name,
                model_version=self.model_version,
                streaming=False,
                api_key=self._api_key,
                project_id=self._project_id,
            )
        return self._client

    def translate_text_prompt(self, prompt: TextPrompt) -> MistralRequest:
        return MistralRequest(model=self.model_name, messages=[{"role": _USER_ROLE, "content": prompt.text}])

    def evaluate(self, request: MistralRequest) -> MistralResponse:
        response = self.client.request(request.model_dump(exclude_none=True))  # type: ignore
        return MistralResponse(**response)

    def translate_response(self, request: MistralRequest, response: MistralResponse) -> SUTResponse:
        completions = []
        for choice in response.choices:
            text = choice["message"]["content"]
            assert text is not None
            completions.append(SUTCompletion(text=text))
        return SUTResponse(completions=completions)


VERTEX_KEY = InjectSecret(VertexAIAPIKey)
VERTEX_PROJECT_ID = InjectSecret(VertexAIProjectId)

model = "mistral-large"
model_version = "2407"
SUTS.register(MistralAISut, model, model, model_version, VERTEX_KEY, VERTEX_PROJECT_ID)


# model = "ministral-8b-instruct"
# model_version = "2410"
# SUTS.register(MistralAISut, model, model, model_version, VERTEX_KEY, VERTEX_PROJECT_ID)
