from newhelm.dependency_helper import FromSourceDependencyHelper
from newhelm.prompt import Prompt
from newhelm.tests.simple_safety_tests import SimpleSafetyTests, SSTTestItemContext
from newhelm.single_turn_prompt_response import (
    MeasuredTestItem,
    TestItem,
    PromptWithContext,
)
from newhelm.typed_data import TypedData
from tests.test_dependency_helper import MockExternalData


def _create_mock_dependency_helper(tmpdir, prompts):
    csv_header = ["harm_area", "category", "prompts_final_English"]
    csv_text = "\n".join([",".join(row) for row in [csv_header, *prompts]])
    mock_dependencies = {"prompts_csv": MockExternalData(csv_text)}
    return FromSourceDependencyHelper(
        tmpdir.strpath,
        mock_dependencies,
        required_versions={},
    )


def test_make_test_items(tmpdir):
    text_prompts = [
        [
            '"Physical harm and Violence"',
            "Information and advice seeking",
            "prompt 1",
        ],
        ["Scams and fraud", "Instructions and actions", "prompt 2"],
    ]
    dependency_helper = _create_mock_dependency_helper(tmpdir, text_prompts)
    test = SimpleSafetyTests()
    test_items = test.make_test_items(dependency_helper)

    assert len(test_items) == 2
    assert all([len(item.prompts) == 1 for item in test_items])
    assert all([isinstance(item.context, TypedData) for item in test_items])
    assert all([isinstance(item.context, TypedData) for item in test_items])
    assert test_items[0].prompts[0].prompt.text == "prompt 1"
    assert (
        test_items[0].get_context(SSTTestItemContext).harm_area
        == "Physical harm and Violence"
    )
    assert (
        test_items[0].get_context(SSTTestItemContext).prompt_category
        == "Information and advice seeking"
    )
    assert test_items[1].prompts[0].prompt.text == "prompt 2"
    assert test_items[1].get_context(SSTTestItemContext).harm_area == "Scams and fraud"
    assert (
        test_items[1].get_context(SSTTestItemContext).prompt_category
        == "Instructions and actions"
    )


def test_aggregate_measurements(tmpdir):
    test_items = [
        TestItem(
            prompts=[PromptWithContext(prompt=Prompt(text="prompt 1"))],
            context=TypedData.from_instance(
                SSTTestItemContext(
                    harm_area="Physical harm and Violence",
                    prompt_category="Information and advice seeking",
                )
            ),
        ),
        TestItem(
            prompts=[PromptWithContext(prompt=Prompt(text="prompt 2"))],
            context=TypedData.from_instance(
                SSTTestItemContext(
                    harm_area="Scams and fraud",
                    prompt_category="Instructions and actions",
                )
            ),
        ),
    ]

    measured_test_items = [
        MeasuredTestItem(test_item=test_items[0], measurements={"is_safe": 0.0}),
        MeasuredTestItem(test_item=test_items[0], measurements={"is_safe": 1.0}),
    ]

    test = SimpleSafetyTests()
    results = test.aggregate_measurements(measured_test_items)
    result = list(filter(lambda r: r.name == "frac_safe", results))
    assert len(result) == 1  # Should only be 1 result with name frac_safe
    assert result[0].value == 0.5
