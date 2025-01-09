import csv
from abc import ABC
from enum import Enum
from typing import Dict, List, Mapping, Optional

from modelgauge.aggregations import get_measurement_stats_by_key
from modelgauge.annotator_set import AnnotatorSet
from modelgauge.base_test import PromptResponseTest
from modelgauge.default_annotator_set import DEFAULT_ANNOTATOR_SET
from modelgauge.dependency_helper import DependencyHelper
from modelgauge.external_data import ExternalData, WebData
from modelgauge.locales import LOCALES, PUBLIC_LOCALES, validate_locale
from modelgauge.prompt import SUTOptions, TextPrompt
from modelgauge.prompt_sets import ModellabFileDownloadToken, PROMPT_SETS, PromptSet, PromptSetType
from modelgauge.secret_values import InjectSecret
from modelgauge.single_turn_prompt_response import MeasuredTestItem, PromptWithContext, TestItem, TestItemAnnotations
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.test_decorator import modelgauge_test
from modelgauge.test_registry import TESTS
from pydantic import BaseModel


class SafeTestItemContext(BaseModel):
    persona_type: str


class PersonaResult(BaseModel):
    frac_safe: float
    num_items: int


class SafeTestResult(BaseModel):
    persona_results: Mapping[str, PersonaResult]


class SafePersonasVersion1(str, Enum):
    NORMAL = "normal"
    SKILLED = "skilled"
    UNSKILLED = "unskilled"


class BaseSafeTestVersion1(PromptResponseTest, ABC):
    """Version 1.0 of SAFE test. This version is still a work in progress and subject to change.

    Concrete tests must set class attribute `annotators`.
    Assumptions about working group deliverables:
    - Prompt files:
        - Filename format:I
            safe-[hazard uid]-[timestamp]-[supplier code].csv
        - CSV files include columns:
            Prompt UID, Text, Hazard UID, Persona, Locale, Prompt Hash
        - There many be multiple hazards, personas, and/or locales in one file.
        - There may be multiple files associated with each hazard.
          These files differ by supplier and have no overlap in prompts.
    - One hazard per test.
    - One locale per test.
    - Tests can have multiple personas (not 100% certain about this).

    Tests will closely follow the hazard taxonomy used by Llama Guard 3 (https://huggingface.co/meta-llama/Llama-Guard-3-8B#hazard-taxonomy-and-policy).
    Test UIDs follow the form "safe-{hazard acronym}-{locale}-{prompt set}-{version}(-{evaluator})"
    (e.g. safe-vcr-en_us-official-1.0-ensemble). The evaluator part is optional.
    The hazard acronyms are as follows:
    - vcr: Violent Crimes
    - ncr: Non-Violent Crimes
    - src: Sex-Related Crimes
    - cse: Child Sexual Exploitation
    - dfm: Defamation
    - spc: Specialized Advice
    - prv: Privacy Violations
    - ipv: Intellection Property Violations
    - iwp: Indiscriminate Weapons (CBRNE)
    - hte: Hate
    - ssh: Suicide & Self-Harm
    - sxc: Sexual Content
        - sxc_prn: Pornographic
    """

    hazards = ["vcr", "ncr", "src", "cse", "dfm", "spc", "prv", "ipv", "iwp", "hte", "ssh", "sxc"]
    annotators: AnnotatorSet
    TEST_VERSION = "1.0"

    @staticmethod
    def create_uid(hazard: str, locale: str, prompt_set: PromptSet, evaluator=None):
        the_prompt_set = PromptSet.find(prompt_set, PROMPT_SETS)
        assert the_prompt_set is not None
        validate_locale(locale)
        if evaluator is None or evaluator == "default":
            postfix = ""
        else:
            postfix = "-" + evaluator
        uid = f"safe-{hazard}-{locale}-{the_prompt_set.prompt_set_type}-{BaseSafeTestVersion1.TEST_VERSION}{postfix}".lower()
        return uid

    def __init__(
        self,
        uid: str,
        hazard: str,
        locale: str,  # the locale for the test, not the prompt file. The prompt file may include more locales.
        persona_types: List[SafePersonasVersion1],
        prompt_set: PromptSet,
        token: Optional[ModellabFileDownloadToken] = None,
    ):
        self._check_annotators()
        super().__init__(uid)

        assert hazard in self.hazards, f"Invalid hazard type {hazard}."
        validate_locale(locale)
        assert len(persona_types), "At least one persona type must be specified."
        assert len(set(persona_types)) == len(
            persona_types
        ), f"Must specify a unique set of persona types, but got {persona_types}"

        self.prompt_set = PromptSet.find(prompt_set, PROMPT_SETS)
        assert (
            self.prompt_set is not None
        ), f'Invalid prompt set {prompt_set.prompt_set_type}-{prompt_set.locale}. Available prompt sets: \n\t{"\n\t".join(
            [ f"{ps.prompt_set_type}-{ps.locale}" for ps in PROMPT_SETS ]
        )}'

        self.hazard = hazard
        self.locale = locale
        self.persona_types = persona_types
        self.token = token

    @classmethod
    def _check_annotators(cls):
        """Check that the annotator set is defined."""
        if not hasattr(cls, "annotators") or cls.annotators is None:
            raise NotImplementedError("Concrete SafeTestVersion1 classes must set class-attribute `annotators`.")

    def get_dependencies(self) -> Mapping[str, ExternalData]:
        if not self.token:
            headers = None
        else:
            headers = {"auth-token": self.token.value}
        return {
            self.prompt_set.filename: WebData(
                source_url=self.prompt_set.url(),
                headers=headers,
            )
        }

    @classmethod
    def get_annotators(cls) -> List[str]:
        cls._check_annotators()
        return cls.annotators.annotators

    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        sut_options = SUTOptions(
            max_tokens=500,
            temperature=0.01,
        )
        test_items: List[TestItem] = []
        data_file = dependency_helper.get_local_path(self.prompt_set.filename)

        # We're not using a context manager here for better exception handling,
        # so the operator has some insight into why things aren't working.
        try:
            csvfile = open(data_file, "r")
            reader = csv.DictReader(csvfile)
            for row in reader:
                text = row["prompt_text"].strip()
                if not text:
                    continue

                # Check that prompt is for correct hazard/persona/locale.
                hazard = row["hazard"].split("_")[0]
                persona = SafePersonasVersion1(row["persona"])
                locale = row["locale"].lower()
                if hazard != self.hazard:
                    continue
                if persona not in self.persona_types:
                    continue
                if locale != self.locale:
                    continue

                prompt = PromptWithContext(
                    prompt=TextPrompt(text=text, options=sut_options),
                    source_id=row["release_prompt_id"],
                )
                test_items.append(
                    TestItem(
                        prompts=[prompt],
                        context=SafeTestItemContext(persona_type=persona),
                    ),
                )
            if len(test_items) == 0:
                raise RuntimeError(f"No test items created from {data_file}")
        except Exception as exc:
            print(f"Error making test items from {data_file}: {exc}")

        return test_items

    def measure_quality(self, item: TestItemAnnotations) -> Dict[str, float]:
        return self.annotators.evaluate(item)

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


@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
class SafeTestVersion1(BaseSafeTestVersion1):
    annotators = DEFAULT_ANNOTATOR_SET


ALL_PERSONAS = [
    SafePersonasVersion1.NORMAL,
    SafePersonasVersion1.SKILLED,
    SafePersonasVersion1.UNSKILLED,
]


def register_tests(cls, evaluator=None):
    for locale in LOCALES:
        for hazard in cls.hazards:
            for prompt_set in PROMPT_SETS:
                if prompt_set.locale and prompt_set.locale != locale:
                    continue
                test_uid = BaseSafeTestVersion1.create_uid(
                    hazard=hazard, locale=locale, prompt_set=prompt_set, evaluator=evaluator
                )
                # only practice prompt sets in English are publicly available for now
                if prompt_set == PromptSetType.OFFICIAL or locale not in PUBLIC_LOCALES:
                    token = InjectSecret(ModellabFileDownloadToken)
                else:
                    token = None
                TESTS.register(cls, test_uid, hazard, locale, ALL_PERSONAS, prompt_set, token)


# default llama guard annotator, always
register_tests(SafeTestVersion1)


def register_private_annotator_tests(private_annotators, uid_key):
    try:

        @modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
        class PrivateSafeTestVersion1(BaseSafeTestVersion1):
            annotators = private_annotators

        register_tests(PrivateSafeTestVersion1, uid_key)
    except:
        print(f"unexpected failure registering annotators for {uid_key} and {private_annotators}")
        raise
