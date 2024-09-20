import pytest
from unittest.mock import MagicMock
from modelgauge.default_annotator_set import DefaultAnnotatorSet
from modelgauge.suts.together_client import TogetherApiKey


def test_constructor():
    annotators = DefaultAnnotatorSet(
        secrets={"together_api_key": TogetherApiKey("fake")}
    )
    assert len(annotators.annotators) == 1
    assert "llama_guard_2" in annotators.annotators


def test_evaluate():
    annotators = DefaultAnnotatorSet(
        secrets={"together_api_key": TogetherApiKey("fake")}
    )
    item = MagicMock()
    assert type(annotators.evaluate(item).get("is_safe", None)) == float
