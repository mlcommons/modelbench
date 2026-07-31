"""Unit tests for individual ComposerNode subclasses."""

import pytest
from modelgauge_tests.annotator_tests.composer_tests.conftest import (
    DEFAULT_BRANCH,
    FALSE_BRANCH,
    ROUTER_KEY_A,
    ROUTER_KEY_B,
    SCORE1,
    SCORE2,
    TRUE_BRANCH,
    VERDICT_BRANCH,
)
from modelgauge_tests.annotator_tests.composer_tests.mocks import (
    AlwaysTrue,
    AlwaysUnsafe,
    LowerCaser,
    PromptLengthRouter,
    RouterA,
    RouterB,
)

from modelgauge.annotators.composed_annotator import Safety
from modelgauge.annotators.composer.context import EvalContext, NodeOutput
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


def test_router_routes_to_key_a(sample_ctx, router_a):
    output = router_a.run(sample_ctx)
    assert output.value == ROUTER_KEY_A
    assert router_a.next_nodes(output.value) == TRUE_BRANCH


def test_router_routes_to_key_b(sample_ctx, router_b):
    output = router_b.run(sample_ctx)
    assert output.value == ROUTER_KEY_B
    assert router_b.next_nodes(output.value) == FALSE_BRANCH


def test_router_unknown_key_raises(router_a):
    with pytest.raises(KeyError):
        router_a.next_nodes("unknown_key")


def test_router_all_routes_contains_all_branches(router_a):
    routes = router_a.all_routes()
    for target in TRUE_BRANCH:
        assert target in routes
    for target in FALSE_BRANCH:
        assert target in routes


def test_router_all_route_paths_single_group(router_a):
    route_paths = router_a.all_route_paths()
    assert len(route_paths) == 2
    assert list(TRUE_BRANCH) in route_paths
    assert list(FALSE_BRANCH) in route_paths


def test_router_routes_to_verdict(sample_ctx, router_to_verdict):
    output = router_to_verdict.run(sample_ctx)
    assert output.value == ROUTER_KEY_A
    next_nodes = router_to_verdict.next_nodes(output.value)
    assert len(next_nodes) == len(VERDICT_BRANCH)
    assert isinstance(next_nodes[0], Safety)
    assert next_nodes[0].is_safe is True


def test_router_with_two_verdicts_in_path_raises():
    with pytest.raises(ValueError, match="has multiple Verdict routes"):
        RouterA(
            name="bad_router",
            route_map={ROUTER_KEY_A: [Safety(is_safe=True), Safety(is_safe=False)], ROUTER_KEY_B: FALSE_BRANCH},
        )


def test_prompt_length_router_short_prompt(sample_ctx, prompt_length_router):
    # sample_ctx has prompt="Hello, world" (12 chars < 20)
    output = prompt_length_router.run(sample_ctx)
    assert output.value == PromptLengthRouter.SHORT_KEY
    assert prompt_length_router.next_nodes(output.value) == TRUE_BRANCH


def test_prompt_length_router_long_prompt(prompt_length_router):
    ctx = EvalContext(prompt="This is a much longer prompt string", response="Response.")
    output = prompt_length_router.run(ctx)
    assert output.value == PromptLengthRouter.LONG_KEY
    assert prompt_length_router.next_nodes(output.value) == FALSE_BRANCH


def test_gate_with_two_outputs():
    with pytest.raises(ValueError, match="has multiple Verdict routes"):
        AlwaysTrue(
            name="bad_gate",
            routes_true=[Safety(is_safe=True), Safety(is_safe=False)],
            routes_false=FALSE_BRANCH,
        )


def test_note_format_output():
    assert ComposerNode.format_output(3.1415926535) == "3.14"
    assert ComposerNode.format_output("short string") == "short string"
    long_string = "x" * 50
    assert ComposerNode.format_output(long_string) == "x" * 27 + "..."
