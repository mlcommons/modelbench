"""Shared mock node implementations and helpers for evaluator tests."""

import os
import string

import pytest
from modelgauge_tests.annotator_tests.composer_tests.mocks import (
    AlwaysFalse,
    AlwaysSafe,
    AlwaysTrue,
    AlwaysTrueCacheable,
    AlwaysUnsafe,
    BadArbiter,
    FixedScorer,
    LLMEnricher,
    LowerCaser,
    LowerCaseScorer,
    PromptLengthGate,
    PromptLengthRouter,
    RouterA,
    RouterB,
    ThresholdArbiter,
    UnexpectedArbiter,
    UnexpectedOutput,
    UpperCaseScorer,
)

from modelgauge.annotators.composed_annotator import Safety
from modelgauge.annotators.composer.context import EvalContext
from modelgauge.annotators.composer.dag import Composer
from modelgauge.annotators.composer.prompt_enricher import PromptEngineeredNode
from modelgauge.annotators.composer.verdict import Verdict

TRUE_BRANCH: tuple[str | Verdict] = ("true_branch",)
FALSE_BRANCH: tuple[str | Verdict] = ("false_branch",)
DEFAULT_BRANCH: tuple[str | Verdict] = ("next_node",)
BAD_BRANCH: tuple[str | Verdict] = ("undefined_node",)
VERDICT_BRANCH: tuple[str | Verdict] = (Safety(is_safe=True),)
SCORE1 = 1.0
SCORE2 = 2.0
ROUTER_KEY_A = "key_a"
ROUTER_KEY_B = "key_b"

skip_in_ci = pytest.mark.skipif(os.getenv("CI") == "true", reason="skipped in CI")


@pytest.fixture
def always_true_gate() -> AlwaysTrue:
    return AlwaysTrue(name="always_true", route_map={True: TRUE_BRANCH, False: FALSE_BRANCH})


@pytest.fixture
def bad_gate() -> AlwaysTrue:
    return AlwaysTrue(name="bad_gate", route_map={True: BAD_BRANCH, False: FALSE_BRANCH})


@pytest.fixture
def always_false_gate() -> AlwaysFalse:
    return AlwaysFalse(name="always_false", route_map={True: TRUE_BRANCH, False: FALSE_BRANCH})


@pytest.fixture
def router_a() -> RouterA:
    return RouterA(name="router_a", route_map={ROUTER_KEY_A: TRUE_BRANCH, ROUTER_KEY_B: FALSE_BRANCH})


@pytest.fixture
def router_b() -> RouterB:
    return RouterB(name="router_b", route_map={ROUTER_KEY_A: TRUE_BRANCH, ROUTER_KEY_B: FALSE_BRANCH})


@pytest.fixture
def router_to_verdict() -> RouterA:
    return RouterA(name="router_to_verdict", route_map={ROUTER_KEY_A: VERDICT_BRANCH, ROUTER_KEY_B: FALSE_BRANCH})


@pytest.fixture
def prompt_length_router() -> PromptLengthRouter:
    return PromptLengthRouter(
        name="prompt_length_router",
        route_map={PromptLengthRouter.SHORT_KEY: TRUE_BRANCH, PromptLengthRouter.LONG_KEY: FALSE_BRANCH},
    )


@pytest.fixture
def lower_caser() -> LowerCaser:
    return LowerCaser(name="lower_caser", routes=DEFAULT_BRANCH)


@pytest.fixture
def score_1() -> FixedScorer:
    return FixedScorer(name="score_1", value=SCORE1, routes=DEFAULT_BRANCH)


@pytest.fixture
def score_2() -> FixedScorer:
    return FixedScorer(name="score_2", value=SCORE2, routes=DEFAULT_BRANCH)


@pytest.fixture
def costly_enricher() -> LLMEnricher:
    return LLMEnricher(name="costly_enricher", routes=DEFAULT_BRANCH)


@pytest.fixture
def sample_ctx() -> EvalContext:
    return EvalContext(prompt="Hello, world", response="This is a response.")


@pytest.fixture
def always_unsafe() -> AlwaysUnsafe:
    return AlwaysUnsafe(name="always_unsafe")


@pytest.fixture
def always_safe() -> AlwaysSafe:
    return AlwaysSafe(name="always_safe")


@pytest.fixture
def threshold_arbiter() -> ThresholdArbiter:
    return ThresholdArbiter(name="threshold_arbiter", threshold=1.5)


@pytest.fixture
def router_dag() -> Composer:
    """Two-branch DAG: RouterA always picks key_a → AlwaysSafe; key_b → AlwaysUnsafe."""
    return (
        Composer("router_dag", verdict_type=Safety)
        .add_node(
            RouterA(
                name="router",
                route_map={"key_a": ["always_safe"], "key_b": ["always_unsafe"]},
            )
        )
        .add_node(AlwaysSafe(name="always_safe"))
        .add_node(AlwaysUnsafe(name="always_unsafe"))
    )


@pytest.fixture
def one_step_dag():
    return (
        Composer("one_step", verdict_type=Safety)
        .add_node(
            AlwaysFalse(
                name="gate",
                route_map={True: [Safety(is_safe=True)], False: ["always_unsafe"]},
            )
        )
        .add_node(AlwaysUnsafe(name="always_unsafe"))
    )


@pytest.fixture
def cached_minimal_dag(tmp_path):
    return Composer("cached_minimal", verdict_type=Safety, cache_path=tmp_path).add_node(
        AlwaysTrueCacheable(
            name="always_true",
            route_map={True: [Safety(is_safe=True)], False: [Safety(is_safe=False)]},
        )
    )


@pytest.fixture
def cached_simple_dag(tmp_path):
    return (
        Composer("simple_cached", verdict_type=Safety, cache_path=tmp_path)
        .add_node(
            AlwaysTrueCacheable(
                name="always_true",
                route_map={True: ["lower_caser", "prompt_parity"], False: ["always_safe"]},
            )
        )
        .add_node(AlwaysSafe(name="always_safe"))
        .add_node(
            PromptLengthGate(
                name="prompt_parity",
                route_map={True: [Safety(is_safe=False)], False: ["upper_scorer"]},
            )
        )
        .add_node(LowerCaser(name="lower_caser", routes=["lower_scorer", "upper_scorer"]))
        .add_node(LowerCaseScorer(name="lower_scorer", routes=["threshold_arbiter"]))
        .add_node(UpperCaseScorer(name="upper_scorer", routes=["threshold_arbiter"]))
        .add_node(ThresholdArbiter(name="threshold_arbiter", threshold=0.5))
    )


@pytest.fixture
def simple_dag():
    return (
        Composer("simple", verdict_type=Safety)
        .add_node(
            AlwaysTrue(
                name="always_true",
                route_map={True: ["lower_caser", "prompt_parity"], False: ["always_safe"]},
            )
        )
        .add_node(AlwaysSafe(name="always_safe"))
        .add_node(
            PromptLengthGate(
                name="prompt_parity",
                route_map={True: [Safety(is_safe=False)], False: ["upper_scorer"]},
            )
        )
        .add_node(LowerCaser(name="lower_caser", routes=["lower_scorer", "upper_scorer"]))
        .add_node(LowerCaseScorer(name="lower_scorer", routes=["threshold_arbiter"]))
        .add_node(UpperCaseScorer(name="upper_scorer", routes=["threshold_arbiter"]))
        .add_node(ThresholdArbiter(name="threshold_arbiter", threshold=0.5))
    )


@pytest.fixture()
def bad_dag_with_cycle():
    return (
        Composer("cyclic", verdict_type=Safety)
        .add_node(
            AlwaysTrue(
                name="node1",
                route_map={True: ["node2"], False: ["node3"]},
            )
        )
        .add_node(
            AlwaysTrue(
                name="node2",
                route_map={True: ["node3"], False: ["node1"]},
            )
        )
        .add_node(
            AlwaysTrue(
                name="node3",
                route_map={True: [Safety(is_safe=True)], False: [Safety(is_safe=False)]},
            )
        )
    )


@pytest.fixture
def bad_dag_with_undefined_output(simple_dag):
    bad_arbiter = UnexpectedArbiter(name="arbiter")
    simple_dag.add_node(bad_arbiter)
    return simple_dag


@pytest.fixture
def bad_dag_with_bad_arbiter():
    dag = Composer("test", verdict_type=Safety)
    dag.add_node(BadArbiter(name="bad_arbiter"))
    return dag


@pytest.fixture
def bad_one_step_dag():
    return (
        Composer("one_step", verdict_type=Safety)
        .add_node(
            AlwaysFalse(
                name="gate",
                route_map={True: [UnexpectedOutput()], False: ["always_unsafe"]},
            )
        )
        .add_node(AlwaysUnsafe(name="always_unsafe"))
    )


@pytest.fixture
def prompt_enricher() -> PromptEngineeredNode:
    return PromptEngineeredNode(
        name="prompt_enricher",
        routes=["next_node"],
        prompt_template=string.Template("$prompt\n$response"),
        sut_id="demo_yes_no",
    )
