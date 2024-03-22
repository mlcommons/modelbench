import pytest

from newhelm.prompt import TextPrompt
from newhelm.tests.safe import SafeTest, HAZARDS_MAPPING, FOLDER_NAME, SafeTestResult
from newhelm.single_turn_prompt_response import (
    MeasuredTestItem,
    TestItem,
    PromptWithContext,
)
from newhelm.suts.together_client import TogetherApiKey
from tests.fake_dependency_helper import FakeDependencyHelper, make_csv

HAZARDS_LIST = list(HAZARDS_MAPPING.keys())
_HAZARD_1 = HAZARDS_LIST[0]
_HAZARD_2 = HAZARDS_LIST[1]


def _fake_dependency_helper(tmpdir, hazard, rows):
    csv_header = ["Prompt", "Hazard type shorthand", "UID"]
    csv_text = make_csv(csv_header, rows)
    return FakeDependencyHelper(tmpdir, {f"{FOLDER_NAME}_{hazard}_hazardous": csv_text})


def test_make_test_items(tmpdir):
    text_prompts = [
        ["prompt 1", _HAZARD_1, "id1"],
        ["prompt 2", _HAZARD_1, "id2"],
    ]
    test = SafeTest("uid", _HAZARD_1, FOLDER_NAME, TogetherApiKey("some-value"))
    dependency_helper = _fake_dependency_helper(tmpdir, _HAZARD_1, text_prompts)
    test_items = test.make_test_items(dependency_helper)

    assert len(test_items) == 2
    assert all([len(item.prompts) == 1 for item in test_items])
    # Check prompt, while ignoring sut_options
    assert test_items[0].prompts[0].prompt.text == "prompt 1"
    assert test_items[1].prompts[0].prompt.text == "prompt 2"
    # Check source_id
    assert test_items[0].prompts[0].source_id == "id1"
    assert test_items[1].prompts[0].source_id == "id2"


def test_hazard_mismatch_exception(tmpdir):
    """An error is raised if the test encounters an item in the dataset with a different hazard."""
    test = SafeTest("uid", _HAZARD_1, FOLDER_NAME, TogetherApiKey("some-value"))
    dependency_helper = _fake_dependency_helper(
        tmpdir, _HAZARD_1, [["prompt", _HAZARD_2, "id"]]
    )
    with pytest.raises(AssertionError) as err_info:
        test.make_test_items(dependency_helper)
    assert _HAZARD_1 in str(err_info.value)
    assert _HAZARD_2 in str(err_info.value)


def test_different_hazards(tmpdir):
    """Checks that all tests will produce identical TestItems for datasets that only differ in the hazard_name column."""
    items = []
    for hazard in HAZARDS_LIST:
        test = SafeTest("uid", hazard, FOLDER_NAME, TogetherApiKey("some-value"))
        dependency_helper = _fake_dependency_helper(
            tmpdir, hazard, [["prompt", hazard, "id"]]
        )
        items.append(test.make_test_items(dependency_helper)[0])
    assert all(item == items[0] for item in items)


def test_unknown_hazard_exception():
    with pytest.raises(AssertionError, match="Invalid hazard type"):
        SafeTest("uid", "New Hazard", FOLDER_NAME, TogetherApiKey("some-value"))


def test_aggregate_measurements():
    dummy_item = TestItem(
        prompts=[PromptWithContext(prompt=TextPrompt(text="prompt"), source_id="id")]
    )
    # 2 safe, 1 unsafe
    measured_test_items = [
        MeasuredTestItem(test_item=dummy_item, measurements={"is_safe": 0.0}),
        MeasuredTestItem(test_item=dummy_item, measurements={"is_safe": 1.0}),
        MeasuredTestItem(test_item=dummy_item, measurements={"is_safe": 1.0}),
    ]
    test = SafeTest("uid", HAZARDS_LIST[0], FOLDER_NAME, TogetherApiKey("some-value"))
    results = test.aggregate_measurements(measured_test_items)
    assert results == SafeTestResult(frac_safe=2.0 / 3.0)
