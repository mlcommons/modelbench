import http
from abc import ABC
from typing import Optional

import requests

from modelgauge.annotation import EnsembleSafetyAnnotation, SafetyAnnotation
from modelgauge.annotator import Annotator
from modelgauge.annotators.cheval.request import AnnotationRequest
from modelgauge.prompt import ChatPrompt, TextPrompt
from modelgauge.retry_decorator import retry
from modelgauge.secret_values import RequiredSecret, SecretDescription
from modelgauge.sut import SUTResponse


class ChevalAPIKey(RequiredSecret, ABC):
    provider: str = "unspecified"

    @classmethod
    def for_provider(cls, provider):
        cls.provider = provider
        return cls

    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope=cls.provider,
            key="api_key",
            instructions="Ask MLCommons admin for permission.",
        )


class ChevalEndpointUrl(RequiredSecret, ABC):
    provider: str = "unspecified"

    @classmethod
    def for_provider(cls, provider):
        cls.provider = provider
        return cls

    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope=cls.provider,
            key="endpoint_url",
            instructions="Ask MLCommons admin for permission.",
        )


class Cheval:
    def __init__(
        self,
        api_key: str,
        endpoint_url: str,
    ):
        self.api_key = api_key
        self.endpoint_url = endpoint_url

    def knows(self, annotator: str) -> bool:
        annotators = self._make_request(http.HTTPMethod.GET, "annotators")
        return annotator in annotators

    def annotate(self, request: AnnotationRequest) -> SafetyAnnotation:
        response = self._make_request(http.HTTPMethod.POST, "annotations", data=request.model_dump())
        if not isinstance(response, dict):
            raise ValueError(f"Unexpected response type: {type(response)}")
        if "joined_responses" in response:
            return EnsembleSafetyAnnotation(**response)
        return SafetyAnnotation(**response)

    @retry()
    def _make_request(self, method: http.HTTPMethod, path: str, data: Optional[dict] = None):
        response = requests.request(
            method=method,
            url=f"{self.endpoint_url}{path}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=data,
        )
        response.raise_for_status()
        return response.json()


class ChevalAnnotatorError(Exception):
    pass


class ChevalAnnotator(Annotator):
    def __init__(
        self,
        uid: str,
        api_key: ChevalAPIKey,
        endpoint_url: ChevalEndpointUrl,
    ):
        super().__init__(uid)
        self.api_key = api_key.value
        self.endpoint_url = endpoint_url.value
        self.cheval = Cheval(api_key=self.api_key, endpoint_url=self.endpoint_url)
        if not self.cheval.knows(self.uid):
            raise ChevalAnnotatorError(f"Annotator {self.uid} not found at {self.endpoint_url}")

    def translate_prompt(self, prompt: TextPrompt | ChatPrompt, response: SUTResponse):
        if not isinstance(prompt, TextPrompt):
            raise ChevalAnnotatorError("ChevalAnnotator only supports TextPrompt")
        return AnnotationRequest(
            annotator=self.uid,
            prompt=prompt.text,
            response=response.text,
        )

    def annotate(self, request: AnnotationRequest) -> SafetyAnnotation:
        return self.cheval.annotate(request)

    def translate_response(self, request: AnnotationRequest, response: SafetyAnnotation) -> SafetyAnnotation:
        return response
