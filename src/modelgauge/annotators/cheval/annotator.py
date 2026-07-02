import http
import socket
from typing import Optional

import requests
from requests_toolbelt.adapters.socket_options import SocketOptionsAdapter  # type: ignore

from modelgauge.annotation import EnsembleSafetyAnnotation, SafetyAnnotation
from modelgauge.annotators.request import AnnotationRequest
from modelgauge.annotators.sideinfo import SideInformationAwareAnnotator
from modelgauge.retry_decorator import retry
from modelgauge.secret_values import RequiredSecret, SecretDescription

_CHEVAL_SCOPE = "cheval"
_TCP_KEEPALIVE_IDLE_S = 60


class ChevalAPIKey(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope=_CHEVAL_SCOPE,
            key="api_key",
            instructions="Ask MLCommons admin for permission.",
        )


class ChevalEndpointUrl(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope=_CHEVAL_SCOPE,
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
        self._session = requests.Session()
        adapter = SocketOptionsAdapter(socket_options=self._get_socket_options())
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    @staticmethod
    def _get_socket_options() -> list[tuple[int, int, int]]:
        socket_options = list(SocketOptionsAdapter.default_options) + [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
        # Darwin / Linux use different names for the TCP keepalive idle option.
        os_dep_idle_opt = getattr(socket, "TCP_KEEPIDLE", None) or getattr(socket, "TCP_KEEPALIVE", None)
        if os_dep_idle_opt is not None:
            socket_options.append((socket.IPPROTO_TCP, os_dep_idle_opt, _TCP_KEEPALIVE_IDLE_S))
        return socket_options

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
        response = self._session.request(
            method=method,
            url=f"{self.endpoint_url}{path}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=data,
        )
        response.raise_for_status()
        return response.json()


class ChevalAnnotatorError(Exception):
    pass


class ChevalAnnotator(SideInformationAwareAnnotator):
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
            raise ChevalAnnotatorError(f"Annotator {self.uid} not found or not ready at {self.endpoint_url}")

    def annotate(self, request: AnnotationRequest) -> SafetyAnnotation:
        return self.cheval.annotate(request)

    def translate_response(self, request: AnnotationRequest, response: SafetyAnnotation) -> SafetyAnnotation:
        return response
