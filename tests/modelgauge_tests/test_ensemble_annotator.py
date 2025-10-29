import random
import string

import pytest

from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.ensemble_annotator import EnsembleAnnotator
from modelgauge.ensemble_strategies import ENSEMBLE_STRATEGIES
from modelgauge.prompt import TextPrompt
from modelgauge.sut import SUTResponse

from modelgauge_tests.fake_annotator import FakeSafetyAnnotator
from modelgauge_tests.fake_ensemble_strategy import BadEnsembleStrategy, FakeEnsembleStrategy


def generate_uid():
    return "".join(random.choices(string.ascii_lowercase, k=8))


@pytest.fixture
def patched_strategies(monkeypatch):
    monkeypatch.setattr(ANNOTATORS, "_lookup", {}, raising=True)
    monkeypatch.setitem(ENSEMBLE_STRATEGIES, "fake", FakeEnsembleStrategy())
    monkeypatch.setitem(ENSEMBLE_STRATEGIES, "bad", BadEnsembleStrategy())
    return ENSEMBLE_STRATEGIES


@pytest.fixture
def make_ensemble(patched_strategies):  # noqa: ARG001
    def _make(strategy_key, n, annotator_cls):
        uids = [generate_uid() for _ in range(n)]
        for uid in uids:
            ANNOTATORS.register(annotator_cls, uid)
        return EnsembleAnnotator("ens", list(uids), strategy_key)

    return _make


def test_annotate_flow(make_ensemble):
    ens = make_ensemble(strategy_key="fake", n=2, annotator_cls=FakeSafetyAnnotator)
    prompt = TextPrompt(text="hello kitty")
    sut_resp = SUTResponse(text="hello world")

    req = ens.translate_prompt(prompt, sut_resp)
    assert set(req.keys()) == set(ens.annotators.keys())
    for val in req.values():
        assert val.text == "hello world"

    raw = ens.annotate(req)
    assert set(req.keys()) == set(ens.annotators.keys())
    for key, val in raw.items():
        assert val.sut_text == "hello world"
        assert ens.annotators[key].annotate_calls == 1

    out = ens.translate_response(req, raw)
    assert out.is_safe
    assert set(out.joined_responses.keys()) == set(ens.annotators.keys())


def test_missing_annotations_ignored(make_ensemble):
    ens = make_ensemble(strategy_key="fake", n=2, annotator_cls=FakeSafetyAnnotator)
    requests = {"1": "request 1", "2": "request 2"}
    responses = {"1": "response 1"}

    out = ens.translate_response(requests, responses)
    # The fake strategy doesn't care about missing annotations.
    assert out.is_safe


def test_bad_strategy_propagates_error(make_ensemble):
    ens = make_ensemble("bad", 1, FakeSafetyAnnotator)
    prompt = TextPrompt(text="p")
    sut_resp = SUTResponse(text="hi")
    req = ens.translate_prompt(prompt, sut_resp)
    raw = ens.annotate(req)

    with pytest.raises(RuntimeError, match="Failed to compute response"):
        ens.translate_response(req, raw)
