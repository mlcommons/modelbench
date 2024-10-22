import os
from ast import In
from typing import Any, Dict, List, Optional, Union
from xml.parsers.expat import model

import google.auth
import httpx
from google.auth.transport.requests import Request
from modelgauge.config import load_secrets_from_config
from modelgauge.dependency_injection import _replace_with_injected
from modelgauge.general import APIException
from modelgauge.prompt import ChatPrompt, ChatRole, SUTOptions, TextPrompt
from modelgauge.secret_values import InjectSecret, RequiredSecret, SecretDescription
from modelgauge.sut_registry import SUTS

_USER_ROLE = "user"
_ASSISTANT_ROLE = "assistant"

_ROLE_MAP = {
    ChatRole.user: _USER_ROLE,
    ChatRole.sut: _ASSISTANT_ROLE,
}


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
            instructions="See https://cloud.google.com/docs/authentication/api-keys",
        )


class VertexAIClient:
    def __init__(
        self,
        model_name: str,
        model_version: str,
        streaming: bool,
        api_key: VertexAIAPIKey,
        project_id: VertexAIProjectId,
        region: str = os.environ.get("GOOGLE_REGION"),
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.api_key = api_key
        self.project_id = project_id
        self.region = region
        self.streaming = streaming

    def _get_access_token(self) -> str:
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(Request())
        return credentials.token

    def _build_endpoint_url(self) -> str:
        base_url = f"https://{self.region}-aiplatform.googleapis.com/v1/"
        project_fragment = f"projects/{self.project_id}"
        location_fragment = f"locations/{self.region}"
        specifier = "streamRawPredict" if self.streaming else "rawPredict"
        model_fragment = f"publishers/mistralai/models/{self.model_name}@{self.model_version}"
        url = f"{base_url}{'/'.join([project_fragment, location_fragment, model_fragment])}:{specifier}"
        return url

    def _headers(self):
        headers = {
            "Authorization": f"Bearer {self._get_access_token()}",
            "Accept": "application/json",
        }
        return headers

    def request(self, content: str):
        data = {
            "model": model,
            "messages": [{"role": _USER_ROLE, "content": content}],
            "stream": self.streaming,
        }
        with httpx.Client() as client:
            resp = client.post(self._build_endpoint_url(), json=data, headers=self._headers(), timeout=None)
            return resp.text


class MistralAISut:
    def __init__(
        self, uid: str, model_name: str, model_version: str, api_key: VertexAIAPIKey, project_id: VertexAIProjectId
    ):
        super().__init__(uid)
        self.model_name = model_name
        self.client = VertexAIClient(
            model_name=model_name, model_version=model_version, streaming=False, api_key=api_key, project_id=project_id
        )

    def _load_client(self):
        pass

    def translate_text_prompt(self, prompt: TextPrompt):
        pass

    def translate_chat_prompt(self, prompt: ChatPrompt):
        pass

    def _translate_request(self, messages, options: SUTOptions):
        pass

    def evaluate(self, request):
        pass

    def translate_response(self, request, response):
        # {"id":"0c03b86ac4584716827fe243a0948759","object":"chat.completion","created":1729558340,"model":"mistral-large","choices":[{"index":0,"message":{"role":"assistant","content":"Determining the \"best\" French painter can be subjective and depends on personal preferences, as well as the specific criteria one uses to define \"best,\" such as influence, skill, or historical significance. However, several French painters have had a profound impact on the art world. Here are a few notable figures:\n\n1. **Claude Monet (1840-1926)**: Often considered one of the most influential French painters, Monet is a co-founder of French Impressionist painting. His works, such as \"Impression, Sunrise\" and the \"Water Lilies\" series, are renowned worldwide.\n\n2. **Paul Cézanne (1839-1906)**: Cézanne is often referred to as the father of modern art. His innovative approach to form and color significantly influenced 20th-century artists, including Pablo Picasso and Henri Matisse.\n\n3. **Edgar Degas (1834-1917)**: Known for his paintings, sculptures, prints, and drawings, Degas is particularly celebrated for his works depicting dancers. He is considered one of the founders of Impressionism.\n\n4. **Pierre-Auguste Renoir (1841-1919)**: Another leading figure of Impressionism, Renoir is celebrated for his vibrant and sensual paintings, often featuring nudes and scenes of modern life.\n\n5. **Henri Matisse (1869-1954)**: A key figure in modern art, Matisse is known for his use of color and his fluid, original draughtsmanship. He was a leading figure in the Fauvism movement.\n\nEach of these artists has made significant contributions to the world of art, and the \"best\" among them might vary depending on individual tastes and perspectives.","tool_calls":null},"finish_reason":"stop","logprobs":null}],"usage":{"prompt_tokens":10,"total_tokens":426,"completion_tokens":416}}
        pass


model = "mistral-large"
model_version = "2407"
SUTS.register(MistralAISut, model, model, model_version, InjectSecret(VertexAIAPIKey), InjectSecret(VertexAIProjectId))
