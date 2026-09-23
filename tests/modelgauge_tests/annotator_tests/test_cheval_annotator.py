from typing import Any, Dict

import pytest
import requests

from modelgauge.annotation import EnsembleSafetyAnnotation, SafetyAnnotation
from modelgauge.annotators.cheval.annotator import (
    ChevalAnnotator,
    ChevalAnnotatorError,
    ChevalAPIKey,
    ChevalEndpointUrl,
)
from modelgauge.prompt import TextPrompt
from modelgauge.sut import SUTResponse


class _FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200, text: str = ""):
        self._payload = payload
        self.status_code = status_code
        self.headers: Dict[str, str] = {}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self._payload


def _build_annotator(monkeypatch, annotator_uid: str, get_annotators: list[str]):
    post_payload: Dict[str, Any] = {}

    def fake_request(self, method, url, headers=None, json=None):  # type: ignore[override]
        if url.endswith("annotators"):
            return _FakeResponse(get_annotators)
        if url.endswith("annotations"):
            # allow test to set desired payload by mutating post_payload
            return _FakeResponse(post_payload.copy())
        raise AssertionError(f"Unexpected URL in test stub: {url}")

    monkeypatch.setattr(requests.Session, "request", fake_request)

    # Ensure provider is set on the secret classes and instantiate with values
    api_key = ChevalAPIKey("test-api-key")
    endpoint = ChevalEndpointUrl("http://cheval.test/")

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


def patch_responses(monkeypatch, responses: list[_FakeResponse]) -> list[str]:
    correlation_ids: list[str] = []

    def fake_request(self, method, url, headers=None, json=None):  # type: ignore[override]
        correlation_ids.append(headers["X-CORRELATION-ID"])
        return responses.pop(0)

    monkeypatch.setattr(requests.Session, "request", fake_request)
    monkeypatch.setattr("modelgauge.retry_decorator.time.sleep", lambda _: None)
    return correlation_ids


def test_cheval_annotator_retries_share_correlation_id(monkeypatch):
    annotator, _ = _build_annotator(monkeypatch, "dummy", get_annotators=["dummy"])
    correlation_ids = patch_responses(
        monkeypatch,
        [
            _FakeResponse({}, status_code=500),
            _FakeResponse({}, status_code=500),
            _FakeResponse({"is_safe": True, "is_valid": True}),
            _FakeResponse({"is_safe": True, "is_valid": True}),
        ],
    )

    _run_annotation(annotator, "hello")
    _run_annotation(annotator, "dolly")

    assert len(correlation_ids) == 4
    assert correlation_ids[0] == correlation_ids[1] == correlation_ids[2]
    assert correlation_ids[3] != correlation_ids[0]


def test_cheval_annotator_error_includes_correlation_id_and_body(monkeypatch):
    annotator, _ = _build_annotator(monkeypatch, "dummy", get_annotators=["dummy"])
    correlation_ids = patch_responses(
        monkeypatch, [_FakeResponse({}, status_code=500, text="server detail") for _ in range(3)]
    )

    with pytest.raises(requests.HTTPError) as exc_info:
        _run_annotation(annotator, "some completion")

    assert f"correlation_id={correlation_ids[0]}" in str(exc_info.value)
    assert "server detail" in str(exc_info.value)
