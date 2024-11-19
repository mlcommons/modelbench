import os

import google.auth
import httpx
from google.auth.transport.requests import Request

from modelgauge.secret_values import OptionalSecret, RequiredSecret, SecretDescription


class VertexAIAPIKey(RequiredSecret):
    """This key is used to request an access token from Google auth, which is then
    used to call the inference endpoint. It is not used with the endpoint."""

    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="vertexai",
            key="api_key",
            instructions="See https://cloud.google.com/docs/authentication/api-keys",
        )


class VertexAIProjectId(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="vertexai",
            key="project_id",
            instructions="Your Google Cloud Platform project ID.",
        )


class VertexAIRegion(OptionalSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="vertexai",
            key="region",
            instructions="A Google Cloud Platform region.",
        )


class VertexAIClient:
    def __init__(
        self,
        publisher: str,
        model_name: str,
        model_version: str,
        streaming: bool,
        api_key: VertexAIAPIKey,
        project_id: VertexAIProjectId,
        region: VertexAIRegion | str,
    ):
        self.publisher = publisher
        self.model_name = model_name
        self.model_version = model_version
        self.api_key = api_key.value
        self.project_id = project_id.value
        self.streaming = streaming
        if isinstance(region, str):
            self.region = region
        elif isinstance(region, VertexAIRegion):
            self.region = region.value
        else:
            raise ValueError("Incorrect GCP region.")

    def _get_access_token(self) -> str:
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(Request())
        return credentials.token

    def _build_endpoint_url(self) -> str:
        base_url = f"https://{self.region}-aiplatform.googleapis.com/v1/"
        project_fragment = f"projects/{self.project_id}"
        location_fragment = f"locations/{self.region}"
        specifier = "streamRawPredict" if self.streaming else "rawPredict"
        model_fragment = f"publishers/{self.publisher}/models/{self.model_name}@{self.model_version}"
        url = f"{base_url}{'/'.join([project_fragment, location_fragment, model_fragment])}:{specifier}"
        return url

    def _headers(self):
        headers = {
            "Authorization": f"Bearer {self._get_access_token()}",
            "Accept": "application/json",
        }
        return headers

    def request(self, req: dict) -> dict:
        try:
            client = httpx.Client()
            response = client.post(self._build_endpoint_url(), json=req, headers=self._headers(), timeout=None)
            if response.status_code == 200:
                return response.json()
            else:
                raise RuntimeError(f"VertexAI response code {response.status_code}")
        except Exception as exc:
            raise
