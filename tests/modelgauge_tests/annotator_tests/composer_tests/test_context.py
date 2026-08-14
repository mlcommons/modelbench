import pytest

from modelgauge.annotators.composer.context import EvalContext, NodeOutput


def test_prompt_replacement(sample_ctx):
    new_prompt = "New prompt"
    new_ctx = sample_ctx.with_prompt(new_prompt)
    assert new_ctx.prompt == new_prompt
    assert new_ctx.response == sample_ctx.response
    assert new_ctx.metadata == sample_ctx.metadata


def test_response_replacement(sample_ctx):
    new_response = "New response"
    new_ctx = sample_ctx.with_response(new_response)
    assert new_ctx.prompt == sample_ctx.prompt
    assert new_ctx.response == new_response
    assert new_ctx.metadata == sample_ctx.metadata


def test_metadata_replacement(sample_ctx):
    new_metadata = {"key": "value"}
    new_ctx = sample_ctx.with_metadata(new_metadata)
    assert new_ctx.prompt == sample_ctx.prompt
    assert new_ctx.response == sample_ctx.response
    assert new_ctx.metadata == new_metadata


def test_metadata_updates():
    original_metadata = {"key1": "value1"}
    sample_ctx = EvalContext(prompt="Prompt", response="Response", metadata=original_metadata)
    additional_metadata = {"key2": "value2"}
    new_ctx = sample_ctx.with_metadata_updates(additional_metadata)
    assert new_ctx.prompt == sample_ctx.prompt
    assert new_ctx.response == sample_ctx.response
    assert new_ctx.metadata == {**sample_ctx.metadata, **additional_metadata}


def test_with_updates(sample_ctx):
    new_prompt = "Updated prompt"
    new_response = "Updated response"
    new_metadata = {"updated": True}
    new_ctx = sample_ctx.with_updates(
        prompt=new_prompt,
        response=new_response,
        metadata=new_metadata,
    )
    assert new_ctx.prompt == new_prompt
    assert new_ctx.response == new_response
    assert new_ctx.metadata == new_metadata


def test_with_different_parent_prompt_response(sample_ctx):
    parent_outputs = {
        "parent1": NodeOutput(
            value="output1",
            original_ctx=sample_ctx,
            updated_ctx=sample_ctx.with_response("Updated response for parent1"),
        ),
        "parent2": NodeOutput(
            value="output2",
            original_ctx=sample_ctx,
            updated_ctx=sample_ctx.with_prompt("Updated prompt for parent2"),
        ),
    }
    with pytest.raises(
        ValueError,
        match="all parent outputs must have the same updated prompt/response",
    ):
        sample_ctx.with_parent_outputs(parent_outputs)


def test_with_compatible_parent_metadata(sample_ctx):
    parent_outputs = {
        "parent1": NodeOutput(
            value="output1",
            original_ctx=sample_ctx,
            updated_ctx=sample_ctx.with_metadata_updates({"key1": "value1"}),
        ),
        "parent2": NodeOutput(
            value="output2",
            original_ctx=sample_ctx,
            updated_ctx=sample_ctx.with_metadata_updates({"key2": "value2"}),
        ),
    }
    ctx = sample_ctx.with_parent_outputs(parent_outputs)
    assert ctx.metadata == {"key1": "value1", "key2": "value2"}


def test_with_incompatible_parent_metadata(sample_ctx):
    parent_outputs = {
        "parent1": NodeOutput(
            value="output1",
            original_ctx=sample_ctx,
            updated_ctx=sample_ctx.with_metadata_updates({"key1": "value1"}),
        ),
        "parent2": NodeOutput(
            value="output2",
            original_ctx=sample_ctx,
            updated_ctx=sample_ctx.with_metadata_updates({"key1": "value2"}),
        ),
    }
    with pytest.raises(
        ValueError,
        match="Metadata updates cannot have overlapping keys with different values.",
    ):
        sample_ctx.with_parent_outputs(parent_outputs)


def test_parent_output(sample_ctx):
    node_output = NodeOutput(value="result", original_ctx=sample_ctx)
    ctx = sample_ctx.with_parent_outputs({"node_a": node_output})
    assert ctx.parent_output("node_a") is node_output
    assert ctx.parent_output("missing_node") is None


def test_ancestor_output_returns_none_for_unknown_node(sample_ctx):
    parent_out = NodeOutput(value="x", original_ctx=sample_ctx)
    ctx = sample_ctx.with_parent_outputs({"parent": parent_out})
    assert ctx.ancestor_output("nonexistent") is None


def test_ancestor_output_empty_at_root(sample_ctx):
    ctx = sample_ctx.with_parent_outputs({})
    assert ctx.ancestor_outputs() == []


def test_ancestor_output_deep_chain(sample_ctx):
    """great_grandparent → grandparent → parent → child: all three ancestors visible."""
    ggp_out = NodeOutput(value="ggp", original_ctx=sample_ctx)
    ggp_ctx = sample_ctx.with_parent_outputs({"great_grandparent": ggp_out})

    gp_out = NodeOutput(value="gp", original_ctx=ggp_ctx)
    gp_ctx = ggp_ctx.with_parent_outputs({"grandparent": gp_out})

    p_out = NodeOutput(value="p", original_ctx=gp_ctx)
    child_ctx = gp_ctx.with_parent_outputs({"parent": p_out})

    assert child_ctx.ancestor_output("parent") is not None
    assert child_ctx.ancestor_output("grandparent") is not None
    assert child_ctx.ancestor_output("great_grandparent") is not None


def test_eq_with_non_eval_context(sample_ctx):
    assert sample_ctx != "not an EvalContext"


def test_hash(sample_ctx):
    other_ctx = EvalContext(prompt=sample_ctx.prompt, response=sample_ctx.response)
    assert sample_ctx.hash() == other_ctx.hash()

    other_ctx = EvalContext(
        prompt=sample_ctx.prompt,
        response=sample_ctx.response,
        metadata={"key": "value"},
    )
    assert sample_ctx.hash() != other_ctx.hash()

    # Make sure can handle nested dictionaries in metadata
    nested_ctx_1 = EvalContext(
        prompt=sample_ctx.prompt,
        response=sample_ctx.response,
        metadata={"key": "value", "nested": {"key": "value_1"}},
    )
    nested_ctx_2 = EvalContext(
        prompt=sample_ctx.prompt,
        response=sample_ctx.response,
        metadata={"key": "value", "nested": {"key": "value_2"}},
    )
    nested_ctx_2_copy = EvalContext(
        prompt=sample_ctx.prompt,
        response=sample_ctx.response,
        metadata={"key": "value", "nested": {"key": "value_2"}},
    )
    assert nested_ctx_1.hash() != nested_ctx_2.hash()
    assert nested_ctx_2_copy.hash() == nested_ctx_2.hash()
