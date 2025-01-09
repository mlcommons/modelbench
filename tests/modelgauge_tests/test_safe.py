import pytest

from modelgauge.auth.together_key import TogetherApiKey
from modelgauge.locales import EN_US, FR_FR, LOCALES, PUBLIC_LOCALES
from modelgauge.prompt import TextPrompt
from modelgauge.prompt_sets import PROMPT_SET_DEFAULT_VERSION, PROMPT_SETS, PromptSet, PromptSetType
from modelgauge.single_turn_prompt_response import MeasuredTestItem, PromptWithContext, TestItem
from modelgauge.test_registry import TESTS
from modelgauge.tests.safe_v1 import (
    BaseSafeTestVersion1,
    PersonaResult,
    SafePersonasVersion1,
    SafeTestItemContext,
    SafeTestResult,
    SafeTestVersion1,
)

from modelgauge_tests.fake_dependency_helper import FakeDependencyHelper, make_csv

FAKE_TOGETHER_KEY = TogetherApiKey("some-value")


@pytest.fixture
def prompt_set():
    ps = PromptSet(
        prompt_set_type=PromptSetType.PRACTICE,
        locale=EN_US,
        version=PROMPT_SET_DEFAULT_VERSION,
        filename="whatever.csv",
    )
    return ps


def _init_safe_test_v1(prompt_set, hazard, persona_types):
    return SafeTestVersion1(uid="uid", hazard=hazard, locale=EN_US, persona_types=persona_types, prompt_set=prompt_set)


def _init_safe_test_v1_private(prompt_set, hazard, persona_types):
    # TODO: Mock the private annotators
    try:
        return SafeTestVersion1(
            uid="uid", hazard=hazard, locale=EN_US, persona_types=persona_types, prompt_set=prompt_set
        )
    except:
        pass  # TODO: is this what we want?
        # @return _init_safe_test_v1(hazard, persona_types)


@pytest.mark.parametrize(
    "test,hazard,persona_1,persona_2",
    [
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
    def fake_test(self, prompt_set, test, hazard, persona_1, persona_2):
        return test(prompt_set=prompt_set, hazard=hazard, persona_types=[persona_1])

    @pytest.fixture
    def fake_test_multiple_personas(self, prompt_set, test, hazard, persona_1, persona_2):
        return test(prompt_set=prompt_set, hazard=hazard, persona_types=[persona_1, persona_2])

    @staticmethod
    def _make_measured_test_item(persona: SafePersonasVersion1, is_safe: float):
        return MeasuredTestItem(
            test_item=TestItem(
                prompts=[PromptWithContext(prompt=TextPrompt(text=""), source_id="")],
                context=SafeTestItemContext(persona_type=persona),
            ),
            measurements={"is_safe": is_safe},
        )

    def test_unknown_hazard_exception(self, prompt_set, test, hazard, persona_1, persona_2):
        with pytest.raises(AssertionError, match="Invalid hazard type"):
            test(prompt_set, "New Hazard", [persona_1])

    def test_no_specified_personas_exception(self, prompt_set, test, hazard, persona_1, persona_2):
        with pytest.raises(AssertionError, match="At least one persona type must be specified"):
            test(prompt_set, hazard, [])

    # todo: Add test for measure_quality

    def test_aggregate_measurements(self, fake_test, persona_1):
        # 2 safe, 1 unsafe
        measured_test_items = [
            self._make_measured_test_item(persona_1, 0.0),
            self._make_measured_test_item(persona_1, 1.0),
            self._make_measured_test_item(persona_1, 1.0),
        ]
        results = fake_test.aggregate_measurements(measured_test_items)
        assert results == SafeTestResult(persona_results={persona_1: PersonaResult(frac_safe=2.0 / 3.0, num_items=3)})

    def test_aggregate_measurements_multiple_personas(self, fake_test_multiple_personas, persona_1, persona_2):
        # persona_1: 1 unsafe. persona_2: 1 unsafe + 1 safe.
        measured_test_items = [
            self._make_measured_test_item(persona_1, 0.0),
            self._make_measured_test_item(persona_2, 0.0),
            self._make_measured_test_item(persona_2, 1.0),
        ]
        results = fake_test_multiple_personas.aggregate_measurements(measured_test_items)
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
        persona_results = fake_test_multiple_personas.aggregate_measurements(measured_test_items).persona_results
        assert persona_results[persona_1] == persona_results[str(persona_1.value)]
        assert persona_results[persona_2] == persona_results[str(persona_2.value)]


class TestSafeV1:
    hazard = SafeTestVersion1.hazards[0]

    @pytest.fixture
    def safe_test(self):
        ps = PromptSet(prompt_set_type=PromptSetType.PRACTICE, locale="en_us", version="1.0", filename="fake-prompts")
        return _init_safe_test_v1(prompt_set=ps, hazard=self.hazard, persona_types=[SafePersonasVersion1.NORMAL])

    def test_make_test_items_skips_out_of_scope_prompts(self, tmpdir, safe_test):
        csv_header = ["release_prompt_id", "prompt_text", "hazard", "locale", "persona"]
        rows = [
            ["1", "a", self.hazard, EN_US, "normal"],
            ["2", "a", self.hazard, EN_US, "skilled"],
            ["3", "a", self.hazard, FR_FR, "normal"],
            ["4", "a", self.hazard, FR_FR, "skilled"],
        ]
        dependencies = {"fake-prompts": make_csv(csv_header, rows)}
        dependency_helper = FakeDependencyHelper(tmpdir, dependencies)

        safe_test.prompt_set.filename = "fake-prompts"
        test_items = safe_test.make_test_items(dependency_helper)

        assert len(test_items) == 1
        assert test_items[0].prompts[0].source_id == "1"

    def test_make_test_items_skips_bad_hazards(self, tmpdir, safe_test):
        csv_header = ["release_prompt_id", "prompt_text", "hazard", "locale", "persona"]
        rows = [
            ["1", "a", self.hazard, EN_US, SafePersonasVersion1.NORMAL],
            ["2", "a", "wrong", EN_US, SafePersonasVersion1.NORMAL],
        ]
        dependencies = {"fake-prompts": make_csv(csv_header, rows)}
        dependency_helper = FakeDependencyHelper(tmpdir, dependencies)

        safe_test.prompt_set.filename = "fake-prompts"
        test_items = safe_test.make_test_items(dependency_helper)

        assert len(test_items) == 1
        assert test_items[0].prompts[0].source_id == "1"

    def test_default_get_annotators(self, safe_test):
        assert safe_test.get_annotators() == ["llama_guard_2"]

    # TODO: Add this back in after setting up private annotators patches
    # def test_annotators_use_provided_secrets(self, safe_test):
    #     """Make sure annotators are not loading keys from environment."""
    #     from modelgauge.annotators.prompt_engineered_annotator import (  # type: ignore
    #         PromptEngineeredAnnotator,
    #     )
    #
    #     annotators = safe_test.get_annotators()
    #
    #     for annotator in annotators.values():
    #         if isinstance(annotator, PromptEngineeredAnnotator):
    #             assert annotator.config.llm_config.api_key == FAKE_TOGETHER_KEY

    class FakeSafeTest(BaseSafeTestVersion1):
        pass

    def test_error_instantiating_undefined_annotators(self, prompt_set):
        """Instantiating a test that does not define `annotators` raises an error."""
        with pytest.raises(NotImplementedError):
            self.FakeSafeTest("uid", self.hazard, EN_US, [SafePersonasVersion1.NORMAL], prompt_set)

    def test_error_getting_undefined_annotators(self):
        """Calling `get_annotators` on a test that does not define `annotators` raises an error."""
        with pytest.raises(NotImplementedError):
            self.FakeSafeTest.get_annotators()

    def test_registered_tests(self):
        hazards = ["vcr", "ncr", "src", "cse", "dfm", "spc", "prv", "ipv", "iwp", "hte", "ssh", "sxc"]
        for hazard in hazards:
            for locale in PUBLIC_LOCALES:
                for prompt_set in PROMPT_SETS:
                    uid = f"safe-{hazard}-{locale}-{prompt_set.prompt_set_type}-1.0"
                    assert TESTS._lookup.get(uid)

    def test_uid_construction(self):
        # locale/prompt set variants
        ps = PromptSet(prompt_set_type=PromptSetType.OFFICIAL, locale="en_us", version="1.0", filename="")
        assert BaseSafeTestVersion1.create_uid("ncr", EN_US, ps) == "safe-ncr-en_us-official-1.0"
        ps = PromptSet(prompt_set_type=PromptSetType.PRACTICE, locale="fr_fr", version="1.0", filename="")
        assert BaseSafeTestVersion1.create_uid("cse", FR_FR, ps) == "safe-cse-fr_fr-practice-1.0"

        # evaluator variants
        ps = PromptSet(prompt_set_type=PromptSetType.PRACTICE, locale="en_us", version="1.0", filename="")
        assert BaseSafeTestVersion1.create_uid("dfm", EN_US, ps, "default") == "safe-dfm-en_us-practice-1.0"
        assert BaseSafeTestVersion1.create_uid("prv", FR_FR, ps, "ensemble") == "safe-prv-fr_fr-practice-1.0-ensemble"

    @pytest.mark.parametrize("prompt_set", PROMPT_SETS)
    def test_correct_prompt_set_dependency(self, prompt_set):
        practice_test = _init_safe_test_v1(
            prompt_set=prompt_set,
            hazard=self.hazard,
            persona_types=[
                "normal",
            ],
        )
        dependencies = practice_test.get_dependencies()

        assert len(dependencies) == 1

        prompt_set_key = list(dependencies.keys())[0]
        assert prompt_set.filename in prompt_set_key
        # assert prompt_set in dependencies[prompt_set_key].source_url
        assert prompt_set.filename in dependencies[prompt_set_key].source_url
