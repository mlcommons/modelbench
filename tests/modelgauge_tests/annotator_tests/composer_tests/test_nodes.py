"""Unit tests for individual ComposerNode subclasses."""

import pytest
from modelgauge_tests.annotator_tests.composer_tests.conftest import (
    DEFAULT_BRANCH,
    FALSE_BRANCH,
    SCORE1,
    SCORE2,
    TRUE_BRANCH,
)
from modelgauge_tests.annotator_tests.composer_tests.mocks import (
    AlwaysTrue,
    AlwaysUnsafe,
    LowerCaser,
)

from modelgauge.annotators.composed_annotator import Safety
from modelgauge.annotators.composer.context import NodeOutput
from modelgauge.annotators.composer.nodes import ComposerNode


def test_true_routes_to_true_branch(sample_ctx, always_true_gate):
    output = always_true_gate.run(sample_ctx)
    assert output.value
    assert always_true_gate.next_nodes(output.value) == TRUE_BRANCH


def test_false_routes_to_false_branch(sample_ctx, always_false_gate):
    output = always_false_gate.run(sample_ctx)
    assert not output.value
    assert always_false_gate.next_nodes(output.value) == FALSE_BRANCH


def test_lower_caser(sample_ctx, lower_caser):
    output = lower_caser.run(sample_ctx)
    assert output.value == sample_ctx.response.lower()
    assert lower_caser.next_nodes(output.value) == DEFAULT_BRANCH


def test_fixed_scorer(sample_ctx, score_1):
    output = score_1.run(sample_ctx)
    assert output.value == SCORE1
    assert score_1.next_nodes(output.value) == DEFAULT_BRANCH


def test_consistent_arbiters(sample_ctx, score_1, score_2, always_unsafe, always_safe):
    parent_outputs = {
        score_1.name: NodeOutput(value=SCORE1, original_ctx=sample_ctx),
        score_2.name: NodeOutput(value=SCORE2, original_ctx=sample_ctx),
    }
    run_ctx = sample_ctx.with_parent_outputs(parent_outputs)
    output = always_unsafe.run(run_ctx)
    assert output.value.name == "UNSAFE"
    output = always_safe.run(run_ctx)
    assert output.value.name == "SAFE"


def test_threshold_arbiter_true(sample_ctx, threshold_arbiter):
    run_ctx = sample_ctx.with_parent_outputs(
        {
            "parent0": NodeOutput(value=SCORE2, original_ctx=sample_ctx),
            "parent1": NodeOutput(value=SCORE2, original_ctx=sample_ctx),
        }
    )
    output = threshold_arbiter.run(run_ctx)
    assert output.value.name == "UNSAFE"


def test_threshold_arbiter_false(sample_ctx, threshold_arbiter):
    run_ctx = sample_ctx.with_parent_outputs(
        {
            "parent0": NodeOutput(value=SCORE1, original_ctx=sample_ctx),
            "parent1": NodeOutput(value=SCORE1, original_ctx=sample_ctx),
        }
    )
    output = threshold_arbiter.run(run_ctx)
    assert output.value.name == "SAFE"


def test_gate_with_two_outputs():
    with pytest.raises(ValueError, match="has multiple Verdict routes"):
        AlwaysTrue(
            name="bad_gate",
            routes_true=[Safety(is_safe=True), Safety(is_safe=False)],
            routes_false=FALSE_BRANCH,
        )


def test_gate_with_no_true_route():
    with pytest.raises(ValueError, match="requires both routes_true and routes_false"):
        AlwaysTrue(
            name="bad_gate",
            routes_false=FALSE_BRANCH,
        )


def test_gate_with_routes():
    with pytest.raises(ValueError, match="should not have routes"):
        AlwaysTrue(
            name="bad_gate",
            routes_true=TRUE_BRANCH,
            routes_false=FALSE_BRANCH,
            routes=DEFAULT_BRANCH,
        )


def test_enricher_with_binary_routes():
    with pytest.raises(ValueError, match="should not have routes_true= / routes_false="):
        LowerCaser(
            name="bad_enricher",
            routes_true=TRUE_BRANCH,
            routes=DEFAULT_BRANCH,
        )


def test_enricher_with_no_routes():
    with pytest.raises(ValueError, match="requires routes="):
        LowerCaser(
            name="bad_enricher",
        )


def test_arbiter_with_routes():
    with pytest.raises(ValueError, match="is terminal and cannot have routing kwargs"):
        AlwaysUnsafe(
            name="bad_arbiter",
            routes=DEFAULT_BRANCH,
        )


def test_note_format_output():
    assert ComposerNode.format_output(3.1415926535) == "3.14"
    assert ComposerNode.format_output("short string") == "short string"
    long_string = "x" * 50
    assert ComposerNode.format_output(long_string) == "x" * 27 + "..."
