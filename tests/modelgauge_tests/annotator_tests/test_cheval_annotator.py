from typing import Any, Dict

import pytest

from modelgauge.annotation import EnsembleSafetyAnnotation, SafetyAnnotation
from modelgauge.annotators.cheval.annotator import (
    ChevalAnnotator,
    ChevalAnnotatorError,
    ChevalAPIKey,
    ChevalEndpointUrl,
)
from modelgauge.prompt import TextPrompt
from modelgauge.sut import SUTResponse

import requests


class _FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self._payload


def _build_annotator(monkeypatch, annotator_uid: str, get_annotators: list[str]):
    post_payload: Dict[str, Any] = {}

    def fake_request(method, url, headers=None, json=None):  # type: ignore[override]
        if url.endswith("annotators"):
            return _FakeResponse(get_annotators)
        if url.endswith("annotations"):
            # allow test to set desired payload by mutating post_payload
            return _FakeResponse(post_payload.copy())
        raise AssertionError(f"Unexpected URL in test stub: {url}")

    monkeypatch.setattr(requests, "request", fake_request)

    # Ensure provider is set on the secret classes and instantiate with values
    api_key = ChevalAPIKey.for_provider("cheval")("test-api-key")
    endpoint = ChevalEndpointUrl.for_provider("cheval")("http://cheval.test/")

    annotator = ChevalAnnotator(annotator_uid, api_key, endpoint)
    return annotator, post_payload


def _run_annotation(annotator: ChevalAnnotator, completion_text: str) -> SafetyAnnotation:
    prompt = TextPrompt(text="The prompt")
    completion = SUTResponse(text=completion_text)
    request = annotator.translate_prompt(prompt, completion)
    response = annotator.annotate(request)
    return annotator.translate_response(request, response)


def test_cheval_annotator_happy_path_safety(monkeypatch):
    annotator, post_payload = _build_annotator(monkeypatch, "dummy", get_annotators=["dummy"])
    post_payload.update({"is_safe": True, "is_valid": True})

    result = _run_annotation(annotator, "some completion")

    assert isinstance(result, SafetyAnnotation)
    assert result.is_safe is True
    assert result.is_valid is True


def test_cheval_annotator_ensemble_response(monkeypatch):
    annotator, post_payload = _build_annotator(monkeypatch, "dummy", get_annotators=["dummy"])
    post_payload.update(
        {
            "is_safe": True,
            "is_valid": True,
            "joined_responses": {
                "a": {"is_safe": True, "is_valid": True},
                "b": {"is_safe": False, "is_valid": True},
            },
        }
    )

    result = _run_annotation(annotator, "some completion")

    assert isinstance(result, EnsembleSafetyAnnotation)
    assert result.is_safe is True
    assert set(result.joined_responses.keys()) == {"a", "b"}
    assert isinstance(result.joined_responses["a"], SafetyAnnotation)
    assert result.joined_responses["a"].is_safe is True
    assert result.joined_responses["b"].is_safe is False


def test_cheval_annotator_unknown_annotator_raises(monkeypatch):
    with pytest.raises(ChevalAnnotatorError):
        _build_annotator(monkeypatch, "unknown", get_annotators=["dummy"])
