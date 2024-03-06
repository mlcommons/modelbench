from newhelm.prompt import TextPrompt
from newhelm.tests.simple_safety_tests import SimpleSafetyTests, SSTTestItemContext
from newhelm.single_turn_prompt_response import (
    MeasuredTestItem,
    TestItem,
    PromptWithContext,
)
from tests.fake_dependency_helper import FakeDependencyHelper, make_csv


def _fake_dependency_helper(tmpdir, rows):
    csv_header = ["harm_area", "category", "prompts_final_English"]
    csv_text = make_csv(csv_header, rows)
    return FakeDependencyHelper(tmpdir, {"prompts_csv": csv_text})


def test_make_test_items(tmpdir):
    text_prompts = [
        [
            "Physical harm and Violence",
            "Information and advice seeking",
            "prompt 1",
        ],
        ["Scams and fraud", "Instructions and actions", "prompt 2"],
    ]
    dependency_helper = _fake_dependency_helper(tmpdir, text_prompts)
    test = SimpleSafetyTests()
    test_items = test.make_test_items(dependency_helper)

    assert len(test_items) == 2
    assert all([len(item.prompts) == 1 for item in test_items])
    assert all([isinstance(item.context, SSTTestItemContext) for item in test_items])
    assert all([isinstance(item.context, SSTTestItemContext) for item in test_items])
    assert test_items[0].prompts[0].prompt.text == "prompt 1"
    assert test_items[0].context.harm_area == "Physical harm and Violence"
    assert test_items[0].context.prompt_category == "Information and advice seeking"
    assert test_items[1].prompts[0].prompt.text == "prompt 2"
    assert test_items[1].context.harm_area == "Scams and fraud"
    assert test_items[1].context.prompt_category == "Instructions and actions"


def test_aggregate_measurements(tmpdir):
    test_items = [
        TestItem(
            prompts=[PromptWithContext(prompt=TextPrompt(text="prompt 1"))],
            context=SSTTestItemContext(
                harm_area="Physical harm and Violence",
                prompt_category="Information and advice seeking",
            ),
        ),
        TestItem(
            prompts=[PromptWithContext(prompt=TextPrompt(text="prompt 2"))],
            context=SSTTestItemContext(
                harm_area="Scams and fraud",
                prompt_category="Instructions and actions",
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
