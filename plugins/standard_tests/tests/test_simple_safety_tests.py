from modelgauge.prompt import TextPrompt
from modelgauge.single_turn_prompt_response import (
    MeasuredTestItem,
    PromptWithContext,
    TestItem,
)
from modelgauge.suts.together_client import TogetherApiKey
from modelgauge.tests.simple_safety_tests import SimpleSafetyTests, SSTTestItemContext
from tests.fake_dependency_helper import FakeDependencyHelper, make_csv


def _fake_dependency_helper(tmpdir, rows):
    csv_header = ["id", "harm_area", "category", "prompts_final_English"]
    csv_text = make_csv(csv_header, rows)
    return FakeDependencyHelper(tmpdir, {"prompts_csv": csv_text})


def _make_test_item(text, prompt_id, harm_area, category) -> TestItem:
    return TestItem(
        prompts=[PromptWithContext(prompt=TextPrompt(text=text), source_id=prompt_id)],
        context=SSTTestItemContext(harm_area=harm_area, prompt_category=category),
    )


def test_make_test_items(tmpdir):
    text_prompts = [
        [
            "id01",
            "Physical harm and Violence",
            "Information and advice seeking",
            "prompt 1",
        ],
        ["id02", "Scams and fraud", "Instructions and actions", "prompt 2"],
    ]
    dependency_helper = _fake_dependency_helper(tmpdir, text_prompts)
    test = SimpleSafetyTests("uid", TogetherApiKey("some-value"))
    test_items = test.make_test_items(dependency_helper)

    assert len(test_items) == 2
    assert all([len(item.prompts) == 1 for item in test_items])
    assert all([isinstance(item.context, SSTTestItemContext) for item in test_items])
    assert all([isinstance(item.context, SSTTestItemContext) for item in test_items])
    expected_item_1 = _make_test_item(
        "prompt 1",
        "id01",
        "Physical harm and Violence",
        "Information and advice seeking",
    )
    expected_item_2 = _make_test_item(
        "prompt 2", "id02", "Scams and fraud", "Instructions and actions"
    )
    # Ignore the SUTOptions attribute in the prompts
    get_prompts = lambda item: list(map(lambda x: x.prompt.text, item.prompts))
    assert get_prompts(test_items[0]) == get_prompts(expected_item_1)
    assert get_prompts(test_items[1]) == get_prompts(expected_item_2)
    assert test_items[0].context == expected_item_1.context
    assert test_items[1].context == expected_item_2.context


def test_aggregate_measurements(tmpdir):
    test_items = [
        _make_test_item(
            "prompt 1",
            "id01",
            "Physical harm and Violence",
            "Information and advice seeking",
        ),
        _make_test_item(
            "prompt 2", "id02", "Scams and fraud", "Instructions and actions"
        ),
    ]

    measured_test_items = [
        MeasuredTestItem(test_item=test_items[0], measurements={"is_safe": 0.0}),
        MeasuredTestItem(test_item=test_items[0], measurements={"is_safe": 1.0}),
    ]

    test = SimpleSafetyTests("uid", TogetherApiKey("some-value"))
    result = test.aggregate_measurements(measured_test_items)
    assert result.overall.frac_safe == 0.5
