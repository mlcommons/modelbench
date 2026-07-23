from typing import Any

import pytest
import requests

import modelgauge.annotators.safety_dag_registration  # noqa: F401  (register on import)
from modelgauge.annotation import SafetyAnnotation
from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.annotators.cheval.ids import (
    SAFETY_ANNOTATOR_V1_1_1_UID,
    SAFETY_ANNOTATOR_V1_1_UID,
)
from modelgauge.annotators.composed_annotator import (
    AnnotatorArbiter,
    SafetyDAGAnnotator,
)
from modelgauge.annotators.safety_dag import SafetyDAGChevalAnnotator
from modelgauge.prompt import TextPrompt
from modelgauge.sut import SUTResponse


class _FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self._payload


@pytest.fixture
def cheval_stub(monkeypatch):
    state = {"annotation": {"is_safe": True, "is_valid": True}}

    def fake_request(self, method, url, headers=None, json=None):
        if url.endswith("annotators"):
            return _FakeResponse([SAFETY_ANNOTATOR_V1_1_UID])
        if url.endswith("annotations"):
            return _FakeResponse(state["annotation"])
        raise AssertionError(f"Unexpected URL in test stub: {url}")

    monkeypatch.setattr(requests.Session, "request", fake_request)
    return state


def _fake_secrets():
    return {"cheval": {"api_key": "test-key", "endpoint_url": "http://cheval.test/"}}


def test_registry_resolves_safety_dag_annotator(cheval_stub):
    annotator = ANNOTATORS.make_instance(SAFETY_ANNOTATOR_V1_1_1_UID, secrets=_fake_secrets())
    assert isinstance(annotator, SafetyDAGChevalAnnotator)
    assert isinstance(annotator, SafetyDAGAnnotator)
    assert annotator.uid == SAFETY_ANNOTATOR_V1_1_1_UID


def test_safety_dag_has_single_arbiter_node(cheval_stub):
    annotator = ANNOTATORS.make_instance(SAFETY_ANNOTATOR_V1_1_1_UID, secrets=_fake_secrets())
    nodes = annotator.dag._nodes
    assert list(nodes.keys()) == ["cheval_safety"]
    assert isinstance(nodes["cheval_safety"], AnnotatorArbiter)


@pytest.mark.parametrize("is_safe", [True, False])
def test_safety_dag_process_returns_safety_annotation(cheval_stub, is_safe):
    cheval_stub["annotation"] = {"is_safe": is_safe, "is_valid": True}
    annotator = ANNOTATORS.make_instance(SAFETY_ANNOTATOR_V1_1_1_UID, secrets=_fake_secrets())

    result = annotator.process(
        prompt=TextPrompt(text="a prompt"),
        response=SUTResponse(text="a response"),
    )

    assert isinstance(result, SafetyAnnotation)
    assert result.is_safe is is_safe
    assert result.is_valid is True
    assert result.metadata is not None
    assert "cheval_safety" in result.metadata["node_outputs"]
    assert result.metadata["verdict"] == ("SAFE" if is_safe else "UNSAFE")
