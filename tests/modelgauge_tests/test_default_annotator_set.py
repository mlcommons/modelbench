import pytest
from unittest.mock import MagicMock
from modelgauge.default_annotator_set import DefaultAnnotatorSet


def test_constructor():
    annotators = DefaultAnnotatorSet()
    assert len(annotators.annotators) == 1
    assert "llama_guard_2" in annotators.annotators


def test_evaluate():
    annotators = DefaultAnnotatorSet()
    item = MagicMock()
    assert type(annotators.evaluate(item).get("is_safe", None)) == float
