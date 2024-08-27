import pytest

from modelgauge.prompt import TextPrompt
from modelgauge.single_turn_prompt_response import (
    MeasuredTestItem,
    PromptWithContext,
    TestItem,
)
from modelgauge.suts.together_client import TogetherApiKey
from modelgauge.tests.safe import (
    FOLDER_NAME,
    PersonaResult,
    SafePersonas,
    SafeTest,
    SafeTestItemContext,
    SafeTestResult,
)
from modelgauge.tests.safe_v1 import Locale, SafeTestVersion1, SafePersonasVersion1
from tests.fake_dependency_helper import FakeDependencyHelper, make_csv

FAKE_API_KEY = TogetherApiKey("some-value")


def _init_safe_test(hazard, persona_types):
    return SafeTest("uid", hazard, "fake_folder", FAKE_API_KEY, persona_types)


def _init_safe_test_v1(hazard, persona_types):
    return SafeTestVersion1("uid", hazard, Locale.EN_US, FAKE_API_KEY, persona_types)


@pytest.mark.parametrize(
    "test,hazard,persona_1, persona_2",
    [
        (_init_safe_test, "cae", SafePersonas.TYPICAL, SafePersonas.MALICIOUS),
        (
            _init_safe_test_v1,
            "dfm",
            SafePersonasVersion1.NORMAL,
            SafePersonasVersion1.SKILLED,
        ),
    ],
)
class TestSafeTestCommonFunctionalityAcrossVersions:
    @pytest.fixture
    def fake_test(self, test, hazard, persona_1, persona_2):
        return test(hazard, [persona_1])

    @pytest.fixture
    def fake_test_multiple_personas(self, test, hazard, persona_1, persona_2):
        return test(hazard, [persona_1, persona_2])

    @staticmethod
    def _make_measured_test_item(persona: SafePersonas, is_safe: float):
        return MeasuredTestItem(
            test_item=TestItem(
                prompts=[PromptWithContext(prompt=TextPrompt(text=""), source_id="")],
                context=SafeTestItemContext(persona_type=persona),
            ),
            measurements={"is_safe": is_safe},
        )

    def test_unknown_hazard_exception(self, test, hazard, persona_1, persona_2):
        with pytest.raises(AssertionError, match="Invalid hazard type"):
            test("New Hazard", [persona_1])

    def test_no_specified_personas_exception(self, test, hazard, persona_1, persona_2):
        with pytest.raises(
            AssertionError, match="At least one persona type must be specified"
        ):
            test(hazard, [])

    # todo: Add test for measure_quality

    def test_aggregate_measurements(self, fake_test, persona_1):
        # 2 safe, 1 unsafe
        measured_test_items = [
            self._make_measured_test_item(persona_1, 0.0),
            self._make_measured_test_item(persona_1, 1.0),
            self._make_measured_test_item(persona_1, 1.0),
        ]
        results = fake_test.aggregate_measurements(measured_test_items)
        assert results == SafeTestResult(
            persona_results={persona_1: PersonaResult(frac_safe=2.0 / 3.0, num_items=3)}
        )

    def test_aggregate_measurements_multiple_personas(
        self, fake_test_multiple_personas, persona_1, persona_2
    ):
        # persona_1: 1 unsafe. persona_2: 1 unsafe + 1 safe.
        measured_test_items = [
            self._make_measured_test_item(persona_1, 0.0),
            self._make_measured_test_item(persona_2, 0.0),
            self._make_measured_test_item(persona_2, 1.0),
        ]
        results = fake_test_multiple_personas.aggregate_measurements(
            measured_test_items
        )
        assert results == SafeTestResult(
            persona_results={
                persona_1: PersonaResult(frac_safe=0.0, num_items=1),
                persona_2: PersonaResult(frac_safe=0.5, num_items=2),
            }
        )

    def test_result_keyable_with_string_or_enum_persona(
        self, tmpdir, fake_test_multiple_personas, persona_1, persona_2
    ):
        measured_test_items = [
            self._make_measured_test_item(persona_1, 0.0),
            self._make_measured_test_item(persona_2, 1.0),
        ]
        persona_results = fake_test_multiple_personas.aggregate_measurements(
            measured_test_items
        ).persona_results
        assert persona_results[persona_1] == persona_results[str(persona_1.value)]
        assert persona_results[persona_2] == persona_results[str(persona_2.value)]


HAZARDS_LIST = SafeTest.hazards
_HAZARD_1 = HAZARDS_LIST[0]
_HAZARD_2 = HAZARDS_LIST[1]
_PERSONA_1 = SafePersonas.TYPICAL
_PERSONA_2 = SafePersonas.MALICIOUS


def _fake_dependency_helper(tmpdir, hazard, persona_rows_dict):
    csv_header = ["Prompt", "Hazard category shorthand", "UID"]
    dependencies = {}
    for persona, rows in persona_rows_dict.items():
        dependencies[f"{FOLDER_NAME}_{hazard}_{persona.value}"] = make_csv(
            csv_header, rows
        )
    return FakeDependencyHelper(tmpdir, dependencies)


def test_make_test_items(tmpdir):
    text_prompts = [
        ["prompt 1", _HAZARD_1, "id1"],
        ["prompt 2", _HAZARD_1, "id2"],
    ]
    test = SafeTest(
        "uid", _HAZARD_1, FOLDER_NAME, FAKE_API_KEY, persona_types=[_PERSONA_1]
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
        "uid", _HAZARD_1, FOLDER_NAME, FAKE_API_KEY, persona_types=[_PERSONA_1]
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
            "uid", hazard, FOLDER_NAME, FAKE_API_KEY, persona_types=[_PERSONA_1]
        )
        dependency_helper = _fake_dependency_helper(
            tmpdir, hazard, {_PERSONA_1: [["prompt", hazard, "id"]]}
        )
        items.append(test.make_test_items(dependency_helper)[0])
    assert all(item == items[0] for item in items)


def test_different_persona_dependency_keys(tmpdir):
    """Test uses correct dependency key mapping for each persona."""
    dependencies = {}
    for persona in SafePersonas:
        dependencies[persona] = [[f"{persona} prompt", _HAZARD_1, "id1"]]
    dependency_helper = _fake_dependency_helper(tmpdir, _HAZARD_1, dependencies)

    for persona in SafePersonas:
        test = SafeTest(
            "uid", _HAZARD_1, FOLDER_NAME, FAKE_API_KEY, persona_types=[persona]
        )
        item = test.make_test_items(dependency_helper)[0]
        assert item.prompts[0].prompt.text == f"{persona} prompt"


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
        FAKE_API_KEY,
        persona_types=[_PERSONA_1, _PERSONA_2],
    )
    items = test.make_test_items(dependency_helper)
    assert [item.context.persona_type for item in items] == [
        _PERSONA_1,
        _PERSONA_2,
        _PERSONA_2,
    ]
    # Ensure we got all the right prompts, and in a stable order.
    assert [item.prompts[0].prompt.text for item in items] == prompts
