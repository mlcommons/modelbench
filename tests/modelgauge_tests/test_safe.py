import pytest
from unittest.mock import patch
from modelgauge.default_annotator_set import DefaultAnnotatorSet

try:
    from modelgauge.private_ensemble_annotator_set import (
        EnsembleAnnotatorSet,
        HuggingFaceKey,
        VllmApiKey,
    )

    FAKE_HUGGINGFACE_KEY = HuggingFaceKey("fake-hf-token")
    FAKE_VLLM_KEY = VllmApiKey("fake-vllm-key")
except:
    FAKE_HUGGINGFACE_KEY = None  # type: ignore
    FAKE_VLLM_KEY = None  # type: ignore
    pass
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
from modelgauge.tests.safe_v1 import Locale, SafePersonasVersion1, SafeTestVersion1
from tests.fake_dependency_helper import FakeDependencyHelper, make_csv

FAKE_TOGETHER_KEY = TogetherApiKey("some-value")


def _init_safe_test(hazard, persona_types):
    return SafeTest("uid", hazard, "fake_folder", FAKE_TOGETHER_KEY, persona_types)


def _init_safe_test_v1(hazard, persona_types):
    secrets = {"together_api_key": FAKE_TOGETHER_KEY}
    return SafeTestVersion1(
        "uid",
        hazard,
        Locale.EN_US,
        persona_types,
        annotator_class=DefaultAnnotatorSet,
        **secrets,
    )


def _init_safe_test_v1_private(hazard, persona_types):
    # TODO: Mock the private annotators
    try:
        secrets = {
            "together_api_key": FAKE_TOGETHER_KEY,
            "huggingface_key": FAKE_HUGGINGFACE_KEY,
            "vllm_api_key": FAKE_VLLM_KEY,
        }
        return SafeTestVersion1(
            "uid",
            hazard,
            Locale.EN_US,
            persona_types,
            annotator_class=EnsembleAnnotatorSet,
            **secrets,
        )
    except:
        pass  # TODO: is this what we want?
        # @return _init_safe_test_v1(hazard, persona_types)


@pytest.mark.parametrize(
    "test,hazard,persona_1,persona_2",
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


class TestSafeV05:

    hazard_1 = SafeTest.hazards[0]
    persona_1 = SafePersonas.TYPICAL
    persona_2 = SafePersonas.MALICIOUS

    @staticmethod
    def _fake_dependency_helper(tmpdir, hazard, persona_rows_dict):
        csv_header = ["Prompt", "Hazard category shorthand", "UID"]
        dependencies = {}
        for persona, rows in persona_rows_dict.items():
            dependencies[f"{FOLDER_NAME}_{hazard}_{persona.value}"] = make_csv(
                csv_header, rows
            )
        return FakeDependencyHelper(tmpdir, dependencies)

    def test_make_test_items(self, tmpdir):
        text_prompts = [
            ["prompt 1", self.hazard_1, "id1"],
            ["prompt 2", self.hazard_1, "id2"],
        ]
        test = SafeTest(
            "uid",
            self.hazard_1,
            FOLDER_NAME,
            FAKE_TOGETHER_KEY,
            persona_types=[self.persona_1],
        )
        dependency_helper = self._fake_dependency_helper(
            tmpdir, self.hazard_1, {self.persona_1: text_prompts}
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

    def test_hazard_mismatch_exception(self, tmpdir):
        """An error is raised if the test encounters an item in the dataset with a different hazard."""
        hazard_2 = SafeTest.hazards[1]
        test = SafeTest(
            "uid",
            self.hazard_1,
            FOLDER_NAME,
            FAKE_TOGETHER_KEY,
            persona_types=[self.persona_1],
        )
        dependency_helper = self._fake_dependency_helper(
            tmpdir, self.hazard_1, {self.persona_1: [["prompt", hazard_2, "id"]]}
        )
        with pytest.raises(AssertionError) as err_info:
            test.make_test_items(dependency_helper)
        assert self.hazard_1 in str(err_info.value)
        assert hazard_2 in str(err_info.value)

    def test_different_hazards(self, tmpdir):
        """Checks that all tests will produce identical TestItems for datasets that only differ in the hazard_name column."""
        items = []
        for hazard in SafeTest.hazards:
            test = SafeTest(
                "uid",
                hazard,
                FOLDER_NAME,
                FAKE_TOGETHER_KEY,
                persona_types=[self.persona_1],
            )
            dependency_helper = self._fake_dependency_helper(
                tmpdir, hazard, {self.persona_1: [["prompt", hazard, "id"]]}
            )
            items.append(test.make_test_items(dependency_helper)[0])
        assert all(item == items[0] for item in items)

    def test_different_persona_dependency_keys(self, tmpdir):
        """Test uses correct dependency key mapping for each persona."""
        dependencies = {}
        for persona in SafePersonas:
            dependencies[persona] = [[f"{persona} prompt", self.hazard_1, "id1"]]
        dependency_helper = self._fake_dependency_helper(
            tmpdir, self.hazard_1, dependencies
        )

        for persona in SafePersonas:
            test = SafeTest(
                "uid",
                self.hazard_1,
                FOLDER_NAME,
                FAKE_TOGETHER_KEY,
                persona_types=[persona],
            )
            item = test.make_test_items(dependency_helper)[0]
            assert item.prompts[0].prompt.text == f"{persona} prompt"

    def test_multiple_personas_test_items(self, tmpdir):
        prompts = ["prompt 1", "prompt 2", "prompt 3"]
        dependency_helper = self._fake_dependency_helper(
            tmpdir,
            self.hazard_1,
            {
                self.persona_1: [[prompts[0], self.hazard_1, "id1"]],
                self.persona_2: [
                    [prompts[1], self.hazard_1, "id2"],
                    [prompts[2], self.hazard_1, "id3"],
                ],
            },
        )
        test = SafeTest(
            "uid",
            self.hazard_1,
            FOLDER_NAME,
            FAKE_TOGETHER_KEY,
            persona_types=[self.persona_1, self.persona_2],
        )
        items = test.make_test_items(dependency_helper)
        assert [item.context.persona_type for item in items] == [
            self.persona_1,
            self.persona_2,
            self.persona_2,
        ]
        # Ensure we got all the right prompts, and in a stable order.
        assert [item.prompts[0].prompt.text for item in items] == prompts


class TestSafeV1:

    hazard = SafeTestVersion1.hazards[0]

    @pytest.fixture
    def safe_test(self):
        return _init_safe_test_v1(self.hazard, [SafePersonasVersion1.NORMAL])

    def test_make_test_items_skips_out_of_scope_prompts(self, tmpdir, safe_test):
        csv_header = ["prompt_uid", "prompt_text", "hazard", "locale", "persona"]
        rows = [
            ["1", "a", self.hazard, Locale.EN_US, SafePersonasVersion1.NORMAL],
            ["2", "a", self.hazard, Locale.EN_US, SafePersonasVersion1.SKILLED],
            ["3", "a", self.hazard, Locale.FR_FR, SafePersonasVersion1.NORMAL],
            ["4", "a", self.hazard, Locale.FR_FR, SafePersonasVersion1.SKILLED],
        ]
        dependencies = {"fake-prompts": make_csv(csv_header, rows)}
        dependency_helper = FakeDependencyHelper(tmpdir, dependencies)

        safe_test._get_prompt_set_names = lambda: ["fake-prompts"]
        test_items = safe_test.make_test_items(dependency_helper)

        assert len(test_items) == 1
        assert test_items[0].prompts[0].source_id == "1"

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
