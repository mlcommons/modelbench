import string

import pytest

from modelgauge.annotators.composer.context import EvalContext
from modelgauge.annotators.composer.prompt_enricher import PromptEngineeredNode


def test_prompt_enricher_run(prompt_enricher, sample_ctx):
    even_ctx = EvalContext(prompt="this is even", response="with the response")
    output = prompt_enricher.run(even_ctx)
    assert isinstance(output.value, str)
    assert output.value == "Yes"

    odd_ctx = EvalContext(prompt="this is not even", response="with the response")
    output = prompt_enricher.run(odd_ctx)
    assert isinstance(output.value, str)
    assert output.value == "No"


def test_prompt_enricher_bad_template():
    with pytest.raises(
        ValueError,
        match="Prompt template may only have 'prompt' and 'response' placeholders.",
    ):
        PromptEngineeredNode(
            name="bad_enricher",
            routes=["next_node"],
            prompt_template=string.Template("This template has an invalid placeholder: $invalid"),
            sut_id="demo_yes_no",
        )
