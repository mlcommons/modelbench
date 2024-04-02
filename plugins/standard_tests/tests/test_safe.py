import pytest
from newhelm.prompt import TextPrompt
from newhelm.single_turn_prompt_response import (
    MeasuredTestItem,
    PromptWithContext,
    TestItem,
)
from newhelm.suts.together_client import TogetherApiKey
from newhelm.tests.safe import (
    FOLDER_NAME,
    HAZARDS_MAPPING,
    PersonaResult,
    PersonaType,
    SafeTest,
    SafeTestItemContext,
    SafeTestResult,
)
from tests.fake_dependency_helper import FakeDependencyHelper, make_csv

HAZARDS_LIST = list(HAZARDS_MAPPING.keys())
_HAZARD_1 = HAZARDS_LIST[0]
_HAZARD_2 = HAZARDS_LIST[1]
_PERSONA_1 = PersonaType.TYPICAL
_PERSONA_2 = PersonaType.MALICIOUS


def _fake_dependency_helper(tmpdir, hazard, persona_rows_dict):
    csv_header = ["Prompt", "Hazard category shorthand", "UID"]
    dependencies = {}
    for persona, rows in persona_rows_dict.items():
        dependencies[f"{FOLDER_NAME}_{hazard}_{persona}"] = make_csv(csv_header, rows)
    return FakeDependencyHelper(tmpdir, dependencies)


def test_make_test_items(tmpdir):
    text_prompts = [
        ["prompt 1", _HAZARD_1, "id1"],
        ["prompt 2", _HAZARD_1, "id2"],
    ]
    test = SafeTest(
        "uid",
        _HAZARD_1,
        FOLDER_NAME,
        TogetherApiKey("some-value"),
        persona_types=[_PERSONA_1],
    )
    dependency_helper = _fake_dependency_helper(
        tmpdir, _HAZARD_1, {_PERSONA_1: text_prompts}
    )
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
    test = SafeTest(
        "uid",
        _HAZARD_1,
        FOLDER_NAME,
        TogetherApiKey("some-value"),
        persona_types=[_PERSONA_1],
    )
    dependency_helper = _fake_dependency_helper(
        tmpdir, _HAZARD_1, {_PERSONA_1: [["prompt", _HAZARD_2, "id"]]}
    )
    with pytest.raises(AssertionError) as err_info:
        test.make_test_items(dependency_helper)
    assert _HAZARD_1 in str(err_info.value)
    assert _HAZARD_2 in str(err_info.value)


def test_different_hazards(tmpdir):
    """Checks that all tests will produce identical TestItems for datasets that only differ in the hazard_name column."""
    items = []
    for hazard in HAZARDS_LIST:
        test = SafeTest(
            "uid",
            hazard,
            FOLDER_NAME,
            TogetherApiKey("some-value"),
            persona_types=[_PERSONA_1],
        )
        dependency_helper = _fake_dependency_helper(
            tmpdir, hazard, {_PERSONA_1: [["prompt", hazard, "id"]]}
        )
        items.append(test.make_test_items(dependency_helper)[0])
    assert all(item == items[0] for item in items)


def test_different_persona_dependency_keys(tmpdir):
    """Test uses correct dependency key mapping for each persona."""
    dependencies = {}
    for persona in PersonaType:
        dependencies[persona] = [[f"{persona} prompt", _HAZARD_1, "id1"]]
    dependency_helper = _fake_dependency_helper(tmpdir, _HAZARD_1, dependencies)

    for persona in PersonaType:
        test = SafeTest(
            "uid",
            _HAZARD_1,
            FOLDER_NAME,
            TogetherApiKey("some-value"),
            persona_types=[persona],
        )
        item = test.make_test_items(dependency_helper)[0]
        assert item.prompts[0].prompt.text == f"{persona} prompt"


def test_string_personas(tmpdir):
    """Check that valid strings can be used instead of PersonaType enum members."""
    dependency_helper = _fake_dependency_helper(
        tmpdir, _HAZARD_1, {_PERSONA_1: [["prompt", _HAZARD_1, "id"]]}
    )
    string_persona_test = SafeTest(
        "uid",
        _HAZARD_1,
        FOLDER_NAME,
        TogetherApiKey("some-value"),
        persona_types=["typical"],
    )
    enum_persona_test = SafeTest(
        "uid",
        _HAZARD_1,
        FOLDER_NAME,
        TogetherApiKey("some-value"),
        persona_types=[PersonaType.TYPICAL],
    )
    assert string_persona_test.persona_types == enum_persona_test.persona_types
    assert (
        string_persona_test.make_test_items(dependency_helper)[0]
        == enum_persona_test.make_test_items(dependency_helper)[0]
    )


def test_multiple_personas_test_items(tmpdir):
    prompts = ["prompt 1", "prompt 2", "prompt 3"]
    dependency_helper = _fake_dependency_helper(
        tmpdir,
        _HAZARD_1,
        {
            _PERSONA_1: [[prompts[0], _HAZARD_1, "id1"]],
            _PERSONA_2: [
                [prompts[1], _HAZARD_1, "id2"],
                [prompts[2], _HAZARD_1, "id3"],
            ],
        },
    )
    test = SafeTest(
        "uid",
        _HAZARD_1,
        FOLDER_NAME,
        TogetherApiKey("some-value"),
        persona_types=[_PERSONA_1, _PERSONA_2],
    )
    items = test.make_test_items(dependency_helper)
    assert {
        (item.prompts[0].prompt.text, item.context.persona_type) for item in items
    } == set(zip(prompts, [_PERSONA_1, _PERSONA_2, _PERSONA_2]))


def test_unknown_hazard_exception():
    with pytest.raises(AssertionError, match="Invalid hazard type"):
        SafeTest(
            "uid",
            "New Hazard",
            FOLDER_NAME,
            TogetherApiKey("some-value"),
            persona_types=[_PERSONA_1],
        )


def test_no_specified_personas_exception():
    with pytest.raises(
        AssertionError, match="At least one persona type must be specified"
    ):
        SafeTest(
            "uid",
            _HAZARD_1,
            FOLDER_NAME,
            TogetherApiKey("some-value"),
            persona_types=[],
        )


def test_unknown_persona_runtime_exception(tmpdir):
    invalid_persona = "new persona"
    dependency_helper = _fake_dependency_helper(
        tmpdir, _HAZARD_1, {_PERSONA_1: [["prompt1", _HAZARD_1, "id1"]]}
    )
    test = SafeTest(
        "uid",
        _HAZARD_1,
        FOLDER_NAME,
        TogetherApiKey("some-value"),
        persona_types=[_PERSONA_1, invalid_persona],
    )
    with pytest.raises(AssertionError) as err_info:
        test.make_test_items(dependency_helper)
        assert invalid_persona in str(err_info.value)


def test_aggregate_measurements():
    dummy_item = TestItem(
        prompts=[PromptWithContext(prompt=TextPrompt(text="prompt"), source_id="id")],
        context=SafeTestItemContext(persona_type=_PERSONA_1),
    )
    # 2 safe, 1 unsafe
    measured_test_items = [
        MeasuredTestItem(test_item=dummy_item, measurements={"is_safe": 0.0}),
        MeasuredTestItem(test_item=dummy_item, measurements={"is_safe": 1.0}),
        MeasuredTestItem(test_item=dummy_item, measurements={"is_safe": 1.0}),
    ]
    test = SafeTest(
        "uid",
        _HAZARD_1,
        FOLDER_NAME,
        TogetherApiKey("some-value"),
        persona_types=[_PERSONA_1],
    )
    results = test.aggregate_measurements(measured_test_items)
    assert results == SafeTestResult(
        persona_results={_PERSONA_1: PersonaResult(frac_safe=2.0 / 3.0, num_items=3)}
    )


def test_aggregate_measurements_multiple_personas():
    dummy_item_persona_1 = TestItem(
        prompts=[PromptWithContext(prompt=TextPrompt(text="prompt"), source_id="id")],
        context=SafeTestItemContext(persona_type=_PERSONA_1),
    )
    dummy_item_persona_2 = TestItem(
        prompts=[PromptWithContext(prompt=TextPrompt(text="prompt"), source_id="id")],
        context=SafeTestItemContext(persona_type=_PERSONA_2),
    )
    # 2 safe, 1 unsafe
    measured_test_items = [
        MeasuredTestItem(test_item=dummy_item_persona_1, measurements={"is_safe": 0.0}),
        MeasuredTestItem(test_item=dummy_item_persona_2, measurements={"is_safe": 0.0}),
        MeasuredTestItem(test_item=dummy_item_persona_2, measurements={"is_safe": 1.0}),
    ]
    test = SafeTest(
        "uid",
        _HAZARD_1,
        FOLDER_NAME,
        TogetherApiKey("some-value"),
        persona_types=[_PERSONA_1, _PERSONA_2],
    )
    results = test.aggregate_measurements(measured_test_items)
    assert results == SafeTestResult(
        persona_results={
            _PERSONA_1: PersonaResult(frac_safe=0.0, num_items=1),
            _PERSONA_2: PersonaResult(frac_safe=0.5, num_items=2),
        }
    )


def test_result_keyable_with_string_or_enum_persona(tmpdir):
    measured_test_items = [
        MeasuredTestItem(
            test_item=TestItem(
                prompts=[
                    PromptWithContext(
                        prompt=TextPrompt(text="typical prompt"), source_id="id1"
                    )
                ],
                context=SafeTestItemContext(persona_type=PersonaType.TYPICAL),
            ),
            measurements={"is_safe": 0.0},
        ),
        MeasuredTestItem(
            test_item=TestItem(
                prompts=[
                    PromptWithContext(
                        prompt=TextPrompt(text="malicious prompt"), source_id="id2"
                    )
                ],
                context=SafeTestItemContext(persona_type=PersonaType.MALICIOUS),
            ),
            measurements={"is_safe": 1.0},
        ),
    ]
    test = SafeTest(
        "uid",
        _HAZARD_1,
        FOLDER_NAME,
        TogetherApiKey("some-value"),
        persona_types=[PersonaType.TYPICAL, PersonaType.MALICIOUS],
    )
    persona_results = test.aggregate_measurements(measured_test_items).persona_results
    assert persona_results[PersonaType.TYPICAL] == persona_results["typical"]
    assert persona_results[PersonaType.MALICIOUS] == persona_results["malicious"]
