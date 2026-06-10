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
SCORE1 = 1.0
SCORE2 = 2.0

skip_in_ci = pytest.mark.skipif(os.getenv("CI") == "true", reason="skipped in CI")


@pytest.fixture
def always_true_gate() -> AlwaysTrue:
    return AlwaysTrue(name="always_true", routes_true=TRUE_BRANCH, routes_false=FALSE_BRANCH)


@pytest.fixture
def bad_gate() -> AlwaysTrue:
    return AlwaysTrue(name="bad_gate", routes_true=BAD_BRANCH, routes_false=FALSE_BRANCH)


@pytest.fixture
def always_false_gate() -> AlwaysFalse:
    return AlwaysFalse(name="always_false", routes_true=TRUE_BRANCH, routes_false=FALSE_BRANCH)


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
def one_step_dag():
    return (
        Composer("one_step", verdict_type=Safety)
        .add_node(
            AlwaysFalse(
                name="gate",
                routes_true=[Safety(is_safe=True)],
                routes_false=["always_unsafe"],
            )
        )
        .add_node(AlwaysUnsafe(name="always_unsafe"))
    )


@pytest.fixture
def cached_minimal_dag(tmp_path):
    return Composer("cached_minimal", verdict_type=Safety, cache_path=tmp_path).add_node(
        AlwaysTrueCacheable(
            name="always_true",
            routes_true=[Safety(is_safe=True)],
            routes_false=[Safety(is_safe=False)],
        )
    )


@pytest.fixture
def cached_simple_dag(tmp_path):
    return (
        Composer("simple_cached", verdict_type=Safety, cache_path=tmp_path)
        .add_node(
            AlwaysTrueCacheable(
                name="always_true",
                routes_true=["lower_caser", "prompt_parity"],
                routes_false=["always_safe"],
            )
        )
        .add_node(AlwaysSafe(name="always_safe"))
        .add_node(
            PromptLengthGate(
                name="prompt_parity",
                routes_true=[Safety(is_safe=False)],
                routes_false=["upper_scorer"],
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
                routes_true=["lower_caser", "prompt_parity"],
                routes_false=["always_safe"],
            )
        )
        .add_node(AlwaysSafe(name="always_safe"))
        .add_node(
            PromptLengthGate(
                name="prompt_parity",
                routes_true=[Safety(is_safe=False)],
                routes_false=["upper_scorer"],
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
                routes_true=["node2"],
                routes_false=["node3"],
            )
        )
        .add_node(
            AlwaysTrue(
                name="node2",
                routes_true=["node3"],
                routes_false=["node1"],
            )
        )
        .add_node(
            AlwaysTrue(
                name="node3",
                routes_true=[Safety(is_safe=True)],
                routes_false=[Safety(is_safe=False)],
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
                routes_true=[UnexpectedOutput()],
                routes_false=["always_unsafe"],
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
