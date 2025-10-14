import pytest

from modelgauge.annotation import SafetyAnnotation
from modelgauge.auth.together_key import TogetherApiKey
from modelgauge.base_test import MissingAnnotationError
from modelgauge.locales import EN_US, FR_FR, LOCALES
from modelgauge.prompt import TextPrompt
from modelgauge.prompt_sets import GENERAL_PROMPT_SETS, prompt_set_to_filename  # usort: skip
from modelgauge.single_turn_prompt_response import MeasuredTestItem, SUTResponseAnnotations, TestItem
from modelgauge.sut import SUTResponse
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


def _init_safe_test_v1(hazard, persona_types, prompt_set="demo"):
    return SafeTestVersion1("uid", hazard, EN_US, persona_types, prompt_set)


def _init_safe_test_v1_private(hazard, persona_types, prompt_set="demo"):
    # TODO: Mock the private annotators
    try:
        return SafeTestVersion1("uid", hazard, EN_US, persona_types, prompt_set)
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
    def fake_test(self, test, hazard, persona_1, persona_2):
        return test(hazard, [persona_1])

    @pytest.fixture
    def fake_test_multiple_personas(self, test, hazard, persona_1, persona_2):
        return test(hazard, [persona_1, persona_2])

    @staticmethod
    def _make_measured_test_item(persona: SafePersonasVersion1, is_safe: float):
        return MeasuredTestItem(
            test_item=TestItem(
                prompt=TextPrompt(text=""),
                source_id="",
                context=SafeTestItemContext(persona_type=persona),
            ),
            measurements={"is_safe": is_safe},
        )

    def test_unknown_hazard_exception(self, test, hazard, persona_1, persona_2):
        with pytest.raises(AssertionError, match="Invalid hazard type"):
            test("New Hazard", [persona_1])

    def test_no_specified_personas_exception(self, test, hazard, persona_1, persona_2):
        with pytest.raises(AssertionError, match="At least one persona type must be specified"):
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
        return _init_safe_test_v1(self.hazard, [SafePersonasVersion1.NORMAL])

    def test_make_test_items_skips_out_of_scope_prompts(self, tmpdir, safe_test):
        csv_header = ["release_prompt_id", "prompt_text", "hazard", "locale", "persona"]
        rows = [
            ["1", "a", self.hazard, EN_US, "normal"],
            ["2", "a", self.hazard, EN_US, "skilled"],
            ["3", "a", self.hazard, FR_FR, "normal"],
            ["4", "a", self.hazard, FR_FR, "skilled"],
        ]
        _pps = GENERAL_PROMPT_SETS["demo"][EN_US]
        dependencies = {_pps: make_csv(csv_header, rows)}
        dependency_helper = FakeDependencyHelper(tmpdir, dependencies)

        safe_test.prompt_set_file_name = _pps
        test_items = safe_test.make_test_items(dependency_helper)

        assert len(test_items) == 1
        assert test_items[0].source_id == "1"

    def test_make_test_items_skips_bad_hazards(self, tmpdir, safe_test):
        csv_header = ["release_prompt_id", "prompt_text", "hazard", "locale", "persona"]
        rows = [
            ["1", "a", self.hazard, EN_US, SafePersonasVersion1.NORMAL],
            ["2", "a", "wrong", EN_US, SafePersonasVersion1.NORMAL],
        ]
        pps = GENERAL_PROMPT_SETS["demo"][EN_US]
        dependencies = {pps: make_csv(csv_header, rows)}
        dependency_helper = FakeDependencyHelper(tmpdir, dependencies)

        safe_test.prompt_set_file_name = pps
        test_items = safe_test.make_test_items(dependency_helper)

        assert len(test_items) == 1
        assert test_items[0].source_id == "1"

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

    def make_test_item_annotations(self, annotations):
        test_item = TestItem(prompt=TextPrompt(text="Why did the chicken cross the road?"), source_id="1")
        sut_response = SUTResponse(text="safe")
        return SUTResponseAnnotations(test_item=test_item, sut_response=sut_response, annotations=annotations)

    @pytest.mark.parametrize("is_safe", [True, False])
    def test_measure_quality(self, safe_test, is_safe):
        raw_annotations = {"llama_guard_2": SafetyAnnotation(is_safe=is_safe)}

        evaluation = safe_test.measure_quality(self.make_test_item_annotations(raw_annotations))

        assert evaluation["is_safe"] == float(is_safe)

    def test_measure_quality_empty(self, safe_test):
        raw_annotations = {}
        with pytest.raises(MissingAnnotationError, match="Missing annotation for llama_guard_2"):
            safe_test.measure_quality(self.make_test_item_annotations(raw_annotations))

    def test_registered_tests(self):
        hazards = ["vcr", "ncr", "src", "cse", "dfm", "spc", "prv", "ipv", "iwp", "hte", "ssh", "sxc"]

        for prompt_set, prompt_set_files in GENERAL_PROMPT_SETS.items():
            for locale in prompt_set_files.keys():
                for hazard in hazards:
                    uid = f"safe-{hazard}-{locale}-{prompt_set}-1.1"
                    assert TESTS._lookup.get(uid)

    def test_uid_construction(self):
        # locale/prompt set variants
        assert BaseSafeTestVersion1.create_uid("xyz", EN_US, "demo") == "safe-xyz-en_us-demo-1.1"
        assert BaseSafeTestVersion1.create_uid("xyz", EN_US, "official") == "safe-xyz-en_us-official-1.1"

        # evaluator variants
        assert BaseSafeTestVersion1.create_uid("xyz", EN_US, "demo", "default") == "safe-xyz-en_us-demo-1.1"
        assert BaseSafeTestVersion1.create_uid("xyz", EN_US, "demo", "official") == "safe-xyz-en_us-demo-1.1-official"

    def test_correct_prompt_set_dependency(self):
        prompt_set = "demo"  # using demo because it doesn't require a token to download
        demo_test = _init_safe_test_v1(self.hazard, "normal", prompt_set=prompt_set)
        dependencies = demo_test.get_dependencies()

        assert len(dependencies) == 1

        prompt_set_key = list(dependencies.keys())[0]
        assert prompt_set_to_filename(prompt_set) in prompt_set_key
        assert prompt_set_to_filename(prompt_set) in dependencies[prompt_set_key].source_url
