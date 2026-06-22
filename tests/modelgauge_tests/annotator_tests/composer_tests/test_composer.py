"""Unit tests for Composer construction, validation, execution, and visualization."""

import json
from unittest.mock import patch

import pandas as pd
import pytest
from modelgauge_tests.annotator_tests.composer_tests.conftest import skip_in_ci
from modelgauge_tests.annotator_tests.composer_tests.mocks import (
    AlwaysSafe,
    AlwaysTrue,
    AlwaysTrueCacheable,
    LowerCaser,
    LowerCaseScorer,
    NoOpEnricher,
    ThresholdArbiter,
    UpperCaser,
)

from modelgauge.annotators.composed_annotator import Safety
from modelgauge.annotators.composer.context import EvalContext
from modelgauge.annotators.composer.dag import Composer, ComposerColumnNames, FailedDAGOutput


def test_dag_outputs(simple_dag):
    assert simple_dag.verdict_type == Safety


def test_dag_with_bad_verdict_type():
    with pytest.raises(
        ValueError,
        match="verdict_type must be a subclass of Verdict",
    ):
        Composer(name="bad_dag", verdict_type=str)


def test_add_node_with_same_name_as_existing_node(simple_dag, always_true_gate):
    always_true_gate.name = next(iter(simple_dag._nodes))
    with pytest.raises(ValueError, match="is already registered"):
        simple_dag.add_node(always_true_gate)  # same name as existing node


def test_add_node_with_undefined_target_node(simple_dag, bad_gate):
    simple_dag.add_node(bad_gate)
    with pytest.raises(ValueError, match="routes to unregistered node"):
        simple_dag._validate_and_build()


def test_dag_with_cycle(bad_dag_with_cycle):
    with pytest.raises(ValueError, match="DAG contains a cycle"):
        bad_dag_with_cycle._validate_and_build()


def test_dag_with_undefined_output(bad_dag_with_undefined_output):
    with pytest.raises(ValueError, match=r"which is not compatible with the DAG\'s verdict_type"):
        bad_dag_with_undefined_output._validate_and_build()


def test_dag_with_bad_arbiter(bad_dag_with_bad_arbiter, sample_ctx):
    with pytest.raises(
        ValueError,
        match=r"DAG execution completed without reaching a Verdict node",
    ):
        bad_dag_with_bad_arbiter.run(sample_ctx)


def test_dag_with_bad_output_route(bad_one_step_dag, sample_ctx):
    with pytest.raises(
        ValueError,
        match=r"incompatible output",
    ):
        bad_one_step_dag.run(sample_ctx)


def test_dag_run_with_cache(cached_minimal_dag, sample_ctx):
    output = cached_minimal_dag.run(sample_ctx)
    assert output.verdict.name == "SAFE"

    always_true = cached_minimal_dag._nodes["always_true"]
    always_true.run = lambda ctx: (_ for _ in ()).throw(ValueError("Should not call run"))
    output_cached = cached_minimal_dag.run(sample_ctx)
    assert output_cached.verdict.name == "SAFE"


def test_dag_uses_per_node_disk_cache(tmp_path, sample_ctx):
    """Each cacheable node must use cache_path / node.name, not a shared store."""
    gate_a = AlwaysTrueCacheable(
        name="gate_a",
        routes_true=["gate_b"],
        routes_false=[Safety(is_safe=False)],
    )
    gate_b = AlwaysTrueCacheable(
        name="gate_b",
        routes_true=[Safety(is_safe=True)],
        routes_false=[Safety(is_safe=False)],
    )
    dag = Composer("per_node_cache", verdict_type=Safety, cache_path=tmp_path).add_node(gate_a).add_node(gate_b)

    assert dag._node_caches["gate_a"].cache_path == tmp_path / "gate_a"
    assert dag._node_caches["gate_b"].cache_path == tmp_path / "gate_b"

    AlwaysTrueCacheable.run_count = 0
    dag.run(sample_ctx)
    assert AlwaysTrueCacheable.run_count == 2

    dag._node_caches["gate_b"].raw_cache.clear()
    gate_a.run = lambda ctx: (_ for _ in ()).throw(ValueError("gate_a cache miss"))

    AlwaysTrueCacheable.run_count = 0
    dag.run(sample_ctx)
    # gate_a hits its own cache; gate_b misses after clear and runs once
    assert AlwaysTrueCacheable.run_count == 1


def test_dag_cache_miss_on_different_context(cached_minimal_dag):
    AlwaysTrueCacheable.run_count = 0
    ctx_a = EvalContext(prompt="Hello", response="world")
    ctx_b = EvalContext(prompt="Goodbye", response="world")

    cached_minimal_dag.run(ctx_a)
    cached_minimal_dag.run(ctx_b)
    assert AlwaysTrueCacheable.run_count == 2


def test_dag_cacheable_node_without_cache_path_runs_each_time(sample_ctx):
    AlwaysTrueCacheable.run_count = 0
    dag = Composer("no_cache", verdict_type=Safety).add_node(
        AlwaysTrueCacheable(
            name="always_true",
            routes_true=[Safety(is_safe=True)],
            routes_false=[Safety(is_safe=False)],
        )
    )
    dag.run(sample_ctx)
    dag.run(sample_ctx)
    assert AlwaysTrueCacheable.run_count == 2


def test_dag_run_with_cached_simple_dag(cached_simple_dag, sample_ctx):
    AlwaysTrueCacheable.run_count = 0

    def assert_output(dag_output):
        assert dag_output.total_cost.input_token_cost == pytest.approx(1.2)
        assert dag_output.total_cost.output_token_cost == pytest.approx(1.6)
        assert dag_output.total_cost.fixed_cost == pytest.approx(0.5)
        assert dag_output.total_cost.latency_seconds == pytest.approx(0.5)
        assert dag_output.total_cost.total_token_cost == pytest.approx(2.8)
        assert dag_output.total_cost.total_cost == pytest.approx(3.3)
        assert dag_output.verdict.name == "UNSAFE"

    dag_output = cached_simple_dag.run(sample_ctx)
    assert_output(dag_output)
    assert AlwaysTrueCacheable.run_count == 1

    dag_output = cached_simple_dag.run(sample_ctx)
    assert_output(dag_output)
    assert AlwaysTrueCacheable.run_count == 1


def test_dag_run(simple_dag, sample_ctx):
    dag_output = simple_dag.run(sample_ctx)

    # lower_caser and prompt_parity are at the same level from always_true
    assert dag_output.total_cost.input_token_cost == pytest.approx(
        1.2
    )  # (lower_caser - 4 * 0.3) + (prompt_parity - 4 * 0)
    assert dag_output.total_cost.output_token_cost == pytest.approx(
        1.6
    )  # (lower_caser - 4 * 0.4) + (prompt_parity - 4 * 0)
    assert dag_output.total_cost.fixed_cost == pytest.approx(0.5)  # (lower_caser - 0.3) + (prompt_parity - 0.2)
    assert dag_output.total_cost.latency_seconds == pytest.approx(0.5)  # (lower_caser - 0.3) + (prompt_parity - 0.2)
    assert dag_output.total_cost.total_token_cost == pytest.approx(2.8)
    assert dag_output.total_cost.total_cost == pytest.approx(3.3)

    assert dag_output.verdict.name == "UNSAFE"


def test_dag_passes_updated_context_to_downstream_nodes():
    ctx = EvalContext(prompt="x", response="HELLO")
    dag = (
        Composer("ctx_update", verdict_type=Safety)
        .add_node(
            AlwaysTrue(
                name="always_true",
                routes_true=["lower_caser"],
                routes_false=["always_safe"],
            )
        )
        .add_node(AlwaysSafe(name="always_safe"))
        .add_node(LowerCaser(name="lower_caser", routes=["noop"]))
        .add_node(NoOpEnricher(name="noop", routes=["lower_scorer"]))
        .add_node(LowerCaseScorer(name="lower_scorer", routes=["threshold_arbiter"]))
        .add_node(ThresholdArbiter(name="threshold_arbiter", threshold=0.5))
    )
    dag_output = dag.run(ctx)
    assert dag_output.node_outputs["lower_caser"].updated_ctx is not None
    assert dag_output.node_outputs["lower_caser"].updated_ctx.response == "hello"
    # Scorer reads ctx.response; 1.0 only if it saw the lowercased update from lower_caser.
    assert dag_output.node_outputs["lower_scorer"].value == pytest.approx(1.0)


def test_dag_updated_context_not_passed_to_parallel_nodes():
    # noop and lower caser are parallel nodes. noop should not see the updated context from lower_caser.
    ctx = EvalContext(prompt="x", response="HELLO")
    dag = (
        Composer("ctx_update", verdict_type=Safety)
        .add_node(
            AlwaysTrue(
                name="always_true",
                routes_true=["lower_caser", "noop"],
                routes_false=["always_safe"],
            )
        )
        .add_node(AlwaysSafe(name="always_safe"))
        .add_node(LowerCaser(name="lower_caser", routes=["lower_scorer"]))
        .add_node(NoOpEnricher(name="noop", routes=["lower_scorer"]))
        .add_node(LowerCaseScorer(name="lower_scorer", routes=["threshold_arbiter"]))
        .add_node(ThresholdArbiter(name="threshold_arbiter", threshold=0.5))
    )
    dag_output = dag.run(ctx)

    assert dag_output.node_outputs["lower_caser"].original_ctx.response == "HELLO"
    assert dag_output.node_outputs["lower_caser"].updated_ctx is not None
    assert dag_output.node_outputs["lower_caser"].updated_ctx.response == "hello"

    assert dag_output.node_outputs["noop"].original_ctx.response == "HELLO"
    assert dag_output.node_outputs["noop"].updated_ctx is None

    assert dag_output.node_outputs["lower_scorer"].original_ctx.response == "hello"
    # Scorer reads ctx.response; 1.0 only if it saw the lowercased update from lower_caser.
    assert dag_output.node_outputs["lower_scorer"].value == pytest.approx(1.0)


def test_dag_output_to_dict(simple_dag, sample_ctx):
    dag_output = simple_dag.run(sample_ctx)
    dag_output_dict = dag_output.to_dict()

    assert "verdict" in dag_output_dict
    assert "total_cost" in dag_output_dict
    assert "node_outputs" in dag_output_dict
    for node_output in dag_output_dict["node_outputs"].values():
        assert "value" in node_output
        assert "original_ctx" in node_output
        assert "updated_ctx" in node_output
        assert "realized_cost" in node_output

    dag_output_dict_no_cost = dag_output.to_dict(skip_cost=True)

    assert "verdict" in dag_output_dict_no_cost
    assert "total_cost" not in dag_output_dict_no_cost
    assert "node_outputs" in dag_output_dict_no_cost
    for node_output in dag_output_dict_no_cost["node_outputs"].values():
        assert "value" in node_output
        assert "original_ctx" in node_output
        assert "updated_ctx" in node_output
        assert "realized_cost" not in node_output


def test_dag_parallel_nodes_different_updated_contexts_raises_error():
    # upper caser and lower caser are parallel nodes, they update the dontext differently which should raise an error.
    ctx = EvalContext(prompt="x", response="HELLO")
    dag = (
        Composer("ctx_update", verdict_type=Safety)
        .add_node(
            AlwaysTrue(
                name="always_true",
                routes_true=["lower_caser", "upper_caser"],
                routes_false=["always_safe"],
            )
        )
        .add_node(AlwaysSafe(name="always_safe"))
        .add_node(LowerCaser(name="lower_caser", routes=["lower_scorer"]))
        .add_node(UpperCaser(name="upper_caser", routes=["lower_scorer"]))
        .add_node(LowerCaseScorer(name="lower_scorer", routes=["threshold_arbiter"]))
        .add_node(ThresholdArbiter(name="threshold_arbiter", threshold=0.5))
    )
    result =dag.run(ctx)
    assert isinstance(result, FailedDAGOutput)


def test_dag_run_with_dataframe_json_md(simple_dag):
    # "hello world" (space lowers avg below threshold) → safe
    # "helloworld"  (no space, avg = 0.5 = threshold)  → unsafe
    # Alternate even/odd prompt lengths to exercise both enricher paths.
    df = pd.DataFrame(
        {
            "prompt": ["a", "ab", "abc", "abcd"],  # odd, even, odd, even
            "response": ["Hello world", "Helloworld", "Hello world", "Helloworld"],
            "metadata": [
                json.dumps({"key": "value1"}),
                json.dumps({"key": "value2"}),
                "notvalidjson",
                None,
            ],
        }
    )
    result_df = simple_dag.run_dataframe(df, metadata_col="metadata")

    assert len(result_df) == len(df)
    assert "prompt" in result_df.columns
    assert "response" in result_df.columns
    verdicts = result_df[simple_dag.df_output_col].tolist()
    expected_verdicts = ["SAFE", "UNSAFE", "SAFE", "UNSAFE"]
    assert verdicts == expected_verdicts

    for cost_json in result_df[simple_dag.df_cost_col]:
        cost = json.loads(cost_json)
        assert "input_token_cost" in cost
        assert "output_token_cost" in cost
        assert "fixed_cost" in cost
        assert "latency_seconds" in cost
        assert "total_token_cost" in cost
        assert "total_cost" in cost

    for dag_run_json in result_df[simple_dag.df_dag_run_col]:
        dag_run = json.loads(dag_run_json)
        assert "always_true" in dag_run
        if "lower_caser" in dag_run:
            # confirm original ctx is present and different from updated ctx
            assert (
                dag_run["lower_caser"]["original_ctx"]["response"].lower()
                == dag_run["lower_caser"]["updated_ctx"]["response"]
            )


def test_dag_run_with_dataframe_cols_md(simple_dag):
    # "hello world" (space lowers avg below threshold) → safe
    # "helloworld"  (no space, avg = 0.5 = threshold)  → unsafe
    # Alternate even/odd prompt lengths to exercise both enricher paths.
    df = pd.DataFrame(
        {
            "prompt": ["a", "ab", "abc", "abcd"],  # odd, even, odd, even
            "response": ["Hello world", "Helloworld", "Hello world", "Helloworld"],
            "hazard": ["low", "high", "low", "high"],
            "other_metadata": ["meta1", "meta2", "meta3", "meta4"],
            "missing_metadata": ["miss1", "miss2", "miss3", "miss4"],
        }
    )
    result_df = simple_dag.run_dataframe(df, metadata_cols=["hazard", "other_metadata"])

    assert len(result_df) == len(df)
    assert "prompt" in result_df.columns
    assert "response" in result_df.columns
    verdicts = result_df[simple_dag.df_output_col].tolist()
    expected_verdicts = ["SAFE", "UNSAFE", "SAFE", "UNSAFE"]
    assert verdicts == expected_verdicts

    for dag_run_info in result_df[simple_dag.df_dag_run_col]:
        dag_run = json.loads(dag_run_info)
        assert "always_true" in dag_run
        assert "original_ctx" in dag_run["always_true"]
        assert "hazard" in dag_run["always_true"]["original_ctx"]["metadata"]
        assert "other_metadata" in dag_run["always_true"]["original_ctx"]["metadata"]
        assert "missing_metadata" not in dag_run["always_true"]["original_ctx"]["metadata"]


def test_dag_run_with_dataframe_cols_md_all(simple_dag):
    # "hello world" (space lowers avg below threshold) → safe
    # "helloworld"  (no space, avg = 0.5 = threshold)  → unsafe
    # Alternate even/odd prompt lengths to exercise both enricher paths.
    df = pd.DataFrame(
        {
            "prompt": ["a", "ab", "abc", "abcd"],  # odd, even, odd, even
            "response": ["Hello world", "Helloworld", "Hello world", "Helloworld"],
            "hazard": ["low", "high", "low", "high"],
            "other_metadata": ["meta1", "meta2", "meta3", "meta4"],
        }
    )
    result_df = simple_dag.run_dataframe(df, metadata_cols=True)

    assert len(result_df) == len(df)
    assert "prompt" in result_df.columns
    assert "response" in result_df.columns
    verdicts = result_df[simple_dag.df_output_col].tolist()
    expected_verdicts = ["SAFE", "UNSAFE", "SAFE", "UNSAFE"]
    assert verdicts == expected_verdicts

    for dag_run_info in result_df[simple_dag.df_dag_run_col]:
        dag_run = json.loads(dag_run_info)
        assert "always_true" in dag_run
        assert "original_ctx" in dag_run["always_true"]
        assert "hazard" in dag_run["always_true"]["original_ctx"]["metadata"]
        assert "other_metadata" in dag_run["always_true"]["original_ctx"]["metadata"]


def test_dag_run_with_dataframe_bad_spec(simple_dag):
    # "hello world" (space lowers avg below threshold) → safe
    # "helloworld"  (no space, avg = 0.5 = threshold)  → unsafe
    # Alternate even/odd prompt lengths to exercise both enricher paths.
    df = pd.DataFrame(
        {
            "prompt": ["a", "ab", "abc", "abcd"],  # odd, even, odd, even
            "response": ["Hello world", "Helloworld", "Hello world", "Helloworld"],
            "hazard": ["low", "high", "low", "high"],
            "other_metadata": ["meta1", "meta2", "meta3", "meta4"],
        }
    )
    with pytest.raises(
        ValueError,
        match="Cannot specify both metadata_col and metadata_cols.",
    ):
        simple_dag.run_dataframe(df, metadata_col="hazard", metadata_cols=True)


def test_dag_run_with_dataframe_parallel(simple_dag):
    df = pd.DataFrame(
        {
            "prompt": ["a", "ab", "abc", "abcd"],  # odd, even, odd, even
            "response": ["hello world", "helloworld", "hello world", "helloworld"],
        }
    )
    result_df = simple_dag.run_dataframe(df, n_jobs=-1)

    assert len(result_df) == len(df)
    assert "prompt" in result_df.columns
    assert "response" in result_df.columns
    verdicts = result_df[simple_dag.df_output_col].tolist()
    expected_verdicts = ["SAFE", "UNSAFE", "SAFE", "UNSAFE"]
    assert verdicts == expected_verdicts


def test_dag_cost_all_paths(simple_dag):
    costs = simple_dag.potential_costs()
    assert costs.keys() == {
        "always_true -> always_safe -> Out (Safety)",
        "always_true -> lower_caser -> prompt_parity -> lower_scorer -> upper_scorer -> threshold_arbiter -> Out (Safety)",
    }

    key = "always_true -> always_safe -> Out (Safety)"
    assert costs[key].input_cost_per_token == pytest.approx(0.0)
    assert costs[key].output_cost_per_token == pytest.approx(0.0)
    assert costs[key].fixed_cost == pytest.approx(1.0)
    assert costs[key].latency_seconds == pytest.approx(1.0)

    key = "always_true -> lower_caser -> prompt_parity -> lower_scorer -> upper_scorer -> threshold_arbiter -> Out (Safety)"
    # 0.3 (lower_caser) + 0.0 (prompt_parity) + 0.0 (lower_scorer) + 0.0 (upper_scorer) + 0.0 (threshold_arbiter)
    assert costs[key].input_cost_per_token == pytest.approx(0.3)
    # 0.4 (lower_caser) + 0.0 (prompt_parity) + 0.0 (lower_scorer) + 0.0 (upper_scorer) + 0.0 (threshold_arbiter)
    assert costs[key].output_cost_per_token == pytest.approx(0.4)
    # 0.3 (lower_caser) + 0.2 (prompt_parity) + 0.7 (lower_scorer) + 0.8 (upper_scorer) + 1.1 (threshold_arbiter)
    assert costs[key].fixed_cost == pytest.approx(3.1)
    # 0.3 (lower_caser) + 0.2 (prompt_parity) + 0.7 (lower_scorer) + 0.8 (upper_scorer) + 1.1 (threshold_arbiter)
    assert costs[key].latency_seconds == pytest.approx(3.1)


@skip_in_ci
def test_dag_visualize_runs(simple_dag, one_step_dag, sample_ctx):
    simple_dag.visualize()
    simple_dag.visualize_run(sample_ctx)
    one_step_dag.visualize()
    one_step_dag.visualize_run(sample_ctx)


def test_visualize_raises_when_graphviz_binary_missing(simple_dag):
    import graphviz

    with patch.object(
        graphviz.Digraph,
        "pipe",
        side_effect=graphviz.ExecutableNotFound(["dot"]),
    ):
        with pytest.raises(
            RuntimeError,
            match="Graphviz system binaries not found",
        ):
            simple_dag.visualize()


def test_composer_names_orig(simple_dag):
    assert simple_dag.name == "simple"
    assert simple_dag.df_output_col == "simple_output"
    assert simple_dag.df_error_col == "simple_error"
    assert simple_dag.df_dag_run_col == "simple_dag_run"
    assert simple_dag.df_cost_col == "simple_dag_cost"


def test_composer_names_override():
    dag = Composer(
        name="dag_name",
        verdict_type=Safety,
        col_names=ComposerColumnNames(composer_name="dag_name", output_col_name="my_output"),
    )
    assert dag.df_output_col == "my_output"
    assert dag.df_error_col == "dag_name_error"
    assert dag.df_dag_run_col == "dag_name_dag_run"
    assert dag.df_cost_col == "dag_name_dag_cost"


def test_composer_names_partial_override_no_name_raises():
    with pytest.raises(ValueError, match="composer_name must be provided"):
        ComposerColumnNames(output_col_name="my_output")
