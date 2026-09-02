from unittest.mock import MagicMock

import pytest

from modelgauge.annotation import SafetyAnnotation
from modelgauge.annotator import Annotator
from modelgauge.annotators.composer.annotator_node import AnnotatorNode
from modelgauge.annotators.composer.context import EvalContext


@pytest.fixture
def annotator():
    a = MagicMock(spec=Annotator)
    a.translate_prompt.return_value = ["fake-request"]
    a.annotate.return_value = "raw-response"
    a.translate_response.return_value = SafetyAnnotation(is_safe=True, is_valid=True)
    a.process.return_value = SafetyAnnotation(is_safe=True, is_valid=True)
    return a


@pytest.fixture
def node(annotator):
    return AnnotatorNode(name="scorer", annotator=annotator, routes=[])


@pytest.fixture
def ctx():
    return EvalContext(prompt="hello", response="what's up")


def test_run_returns_annotation_from_annotator(node, ctx):
    result = node.run(ctx)

    assert isinstance(result.value, SafetyAnnotation)
    assert result.value.is_safe is True
    assert result.value.is_valid is True
    assert result.original_ctx is ctx


def test_run_passes_ctx_prompt_and_response_to_annotator(node, annotator, ctx):
    node.run(ctx)

    test_item_arg, response_arg = annotator.process.call_args.args
    assert test_item_arg.evaluated_prompt.text == "hello"
    assert response_arg.text == "what's up"
    annotator.process.assert_called_once()


def test_run_returns_unsafe_annotation(annotator, ctx):
    annotator.process.return_value = SafetyAnnotation(is_safe=False, is_valid=True)
    node = AnnotatorNode(name="scorer", annotator=annotator, routes=[])

    result = node.run(ctx)
    assert result.value.is_safe is False
    assert result.value.is_valid is True
