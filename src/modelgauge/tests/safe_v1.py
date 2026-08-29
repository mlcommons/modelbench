import csv
from abc import ABC
from typing import Dict, List, Mapping, Optional

from pydantic import BaseModel

from modelgauge.aggregations import get_measurement_stats_by_key
from modelgauge.annotators.cheval.ids import (
    SAFETY_ANNOTATOR_V1_1_UID,
    SAFETY_ANNOTATOR_V1_2_UID,
)
from modelgauge.base_test import PromptResponseTest
from modelgauge.dependency_helper import DependencyHelper
from modelgauge.external_data import ExternalData, WebData
from modelgauge.hazards import Hazards
from modelgauge.locales import validate_locale
from modelgauge.model_options import ModelOptions
from modelgauge.personas import PersonaResult, SafeTestPersonas
from modelgauge.prompt import TextPrompt
from modelgauge.prompt_sets import (  # usort: skip
    GENERAL_PROMPT_SETS,
    PROMPT_SET_DOWNLOAD_URL,
    SAFETY_1_2_PROMPT_SETS,
    ModellabFileDownloadToken,
    prompt_set_file_base_name,
    validate_prompt_set,
    validate_token_requirement,
)
from modelgauge.secret_values import InjectSecret
from modelgauge.single_turn_prompt_response import (
    MeasuredTestItem,
    SUTResponseAnnotations,
    TestItem,
    convert_annotation_to_measurement,
)
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.test_decorator import modelgauge_test
from modelgauge.test_registry import TESTS
from modelgauge.versioned_object import VersionedObject


class SafeTestItemContext(BaseModel):
    persona_type: str
    hazard: str


class SafeTestResult(BaseModel):
    persona_results: Mapping[str, PersonaResult]


class BaseSafeTestVersion1(PromptResponseTest, VersionedObject, ABC):
    """Version 1.x of SAFE test. This version is still a work in progress and subject to change.

    Concrete tests must set class attribute `annotators`.
    Assumptions about working group deliverables:
    - Prompt files:
        - CSV files include columns:
            Prompt UID, Text, Hazard UID, Locale, Persona
        - There many be multiple personas in one file.
        - There is only one locale per file.
        - There may be multiple files associated with each hazard.
          These files differ by supplier and have no overlap in prompts.
    - One hazard per test.
    - One locale per test.
    - Tests can have multiple personas (not 100% certain about this).

    Tests will closely follow the hazard taxonomy used by Llama Guard 3
    (https://huggingface.co/meta-llama/Llama-Guard-3-8B#hazard-taxonomy-and-policy).
    """

    VERSION = "1.x"
    PROMPT_SETS = {}
    hazards = Hazards()
    _sut_options = ModelOptions(
        max_tokens=3000,
        max_total_output_tokens=10000,  # For reasoning SUTs.
        temperature=0.01,
    )

    @classmethod
    def create_uid(cls, hazard: str, locale: str, prompt_set: str, evaluator=None):
        validate_prompt_set(cls.PROMPT_SETS, prompt_set, locale)
        validate_locale(locale)
        if evaluator is None or evaluator == "default":
            suffix = ""
        else:
            suffix = "-" + evaluator
        uid = f"safe-{hazard}-{locale}-{prompt_set}-{cls.VERSION}{suffix}".lower()
        return uid

    def __init__(
        self,
        uid: str,
        hazard: str,
        locale: str,
        persona_types: List[SafeTestPersonas],
        prompt_set: str,
        token: Optional[ModellabFileDownloadToken] = None,
    ):
        super().__init__(uid)
        assert hazard in self.hazards, f"Invalid hazard type {hazard}."
        assert len(persona_types), "At least one persona type must be specified."
        validate_prompt_set(self.PROMPT_SETS, prompt_set, locale)
        validate_token_requirement(prompt_set, token)
        validate_locale(locale)

        self.hazard = hazard
        self.locale = locale
        self.persona_types = list(set(persona_types))
        self.prompt_set_file_base_name = prompt_set_file_base_name(self.PROMPT_SETS, prompt_set, locale)
        self.token = token

    def get_dependencies(self) -> Mapping[str, ExternalData]:
        headers = None
        if self.token is not None:
            headers = {"auth-token": self.token.value}
        source_url = f"{PROMPT_SET_DOWNLOAD_URL}/{self.prompt_set_file_base_name}.csv"
        return {
            self.prompt_set_file_base_name: WebData(
                source_url=source_url,
                headers=headers,
            )
        }

    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        test_items: List[TestItem] = []
        data_file = dependency_helper.get_local_path(self.prompt_set_file_base_name)

        # We're not using a context manager here for better exception handling,
        # so the operator has some insight into why things aren't working.
        csvfile = open(data_file, "r")
        reader = csv.DictReader(csvfile)
        for row in reader:
            text = row["prompt_text"].strip()
            if not text:
                continue

            # Check that prompt is for correct hazard/persona/locale.
            hazard = self.hazards.get_hazard_from_row(row)
            persona = SafeTestPersonas(row["persona"])
            locale = row["locale"].lower()
            if not hazard == self.hazard:
                continue
            if persona not in self.persona_types:
                continue
            if locale != self.locale:
                continue

            visibility = row.get("visibility", "private")
            test_items.append(
                TestItem(
                    prompt=TextPrompt(text=text),
                    source_id=row["release_prompt_id"],
                    context=SafeTestItemContext(persona_type=persona, hazard=hazard),
                    shareable=(visibility == "shareable"),
                ),
            )
        if len(test_items) == 0:
            raise RuntimeError(f"No test items created from {data_file}")

        return test_items

    def measure_quality(self, item: SUTResponseAnnotations) -> Dict[str, float]:
        annotator_uid = self.get_annotators()[0]  # These tests always only have 1 annotator.
        return convert_annotation_to_measurement(item, annotator_uid)

    def aggregate_measurements(self, items: List[MeasuredTestItem]) -> SafeTestResult:
        """Returns a SafeTestResult that maps each persona_type's subset of prompts to its corresponding measurements.

        Persona types that do not have any measured test items are not included.
        """
        measurement_stats = get_measurement_stats_by_key(
            "is_safe", items, key=lambda item: item.test_item.context.persona_type
        )
        persona_results = {}
        for persona in self.persona_types:
            if persona in measurement_stats:
                persona_results[persona.value] = PersonaResult(
                    frac_safe=measurement_stats[persona].mean,
                    num_items=measurement_stats[persona].count,
                )
        return SafeTestResult(persona_results=persona_results)

    @classmethod
    def get_annotators(cls) -> List[str]:
        return ["llama_guard_2"]


@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
class SafeTestVersion1_1(BaseSafeTestVersion1):
    VERSION = "1.1"
    PROMPT_SETS = GENERAL_PROMPT_SETS


@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
class SafeTestVersion1_2(BaseSafeTestVersion1):
    VERSION = "1.2"
    PROMPT_SETS = SAFETY_1_2_PROMPT_SETS


def register_tests(test_class, evaluator=None):
    all_personas = [
        SafeTestPersonas.NORMAL,
        SafeTestPersonas.SKILLED,
        SafeTestPersonas.UNSKILLED,
    ]
    for prompt_set, prompt_set_files in test_class.PROMPT_SETS.items():
        for locale in prompt_set_files.keys():
            for hazard in test_class.hazards:
                test_uid = test_class.create_uid(hazard, locale, prompt_set, evaluator)
                token = InjectSecret(ModellabFileDownloadToken)
                TESTS.register(test_class, test_uid, hazard, locale, all_personas, prompt_set, token)


def register_private_annotator_tests_1_1(private_annotator, evaluator):
    @modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
    class PrivateSafeTestVersion1_1(BaseSafeTestVersion1):
        VERSION = "1.1"
        PROMPT_SETS = GENERAL_PROMPT_SETS

        @classmethod
        def get_annotators(cls) -> List[str]:
            return [private_annotator]

    register_tests(PrivateSafeTestVersion1_1, evaluator)


def register_private_annotator_tests_1_2(private_annotator, evaluator):
    @modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
    class PrivateSafeTestVersion1_2(BaseSafeTestVersion1):
        VERSION = "1.2"
        PROMPT_SETS = SAFETY_1_2_PROMPT_SETS

        @classmethod
        def get_annotators(cls) -> List[str]:
            return [private_annotator]

    register_tests(PrivateSafeTestVersion1_2, evaluator)


# Register tests with "default" annotator.
register_tests(SafeTestVersion1_1)
register_tests(SafeTestVersion1_2)
# Register test with private annotators.
register_private_annotator_tests_1_1(SAFETY_ANNOTATOR_V1_1_UID, "private")
register_private_annotator_tests_1_2(SAFETY_ANNOTATOR_V1_2_UID, "private")
