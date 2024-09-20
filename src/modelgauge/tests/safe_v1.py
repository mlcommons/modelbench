import csv
from enum import Enum
from typing import Dict, List, Mapping, Optional

from modelgauge.aggregations import get_measurement_stats_by_key
from modelgauge.annotator import Annotator
from modelgauge.annotator_set import AnnotatorSet
from modelgauge.base_test import PromptResponseTest
from modelgauge.default_annotator_set import DefaultAnnotatorSet
from modelgauge.dependency_helper import DependencyHelper
from modelgauge.external_data import ExternalData, WebData
from modelgauge.prompt import SUTOptions, TextPrompt
from modelgauge.secret_values import InjectSecret
from modelgauge.single_turn_prompt_response import (
    MeasuredTestItem,
    PromptWithContext,
    TestItem,
    TestItemAnnotations,
)
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.test_decorator import modelgauge_test
from modelgauge.test_registry import TESTS
from modelgauge.tests.safe import PersonaResult, SafeTestItemContext, SafeTestResult


class Locale(str, Enum):
    EN_US = "EN_US"  # English, United States
    FR_FR = "FR_FR"  # French, France
    ZH_CN = "ZH_CN"  # Simplified Chinese, China
    HI_IN = "HI_IN"  # Hindi, India


class SafePersonasVersion1(str, Enum):
    NORMAL = "normal"
    SKILLED = "skilled"
    UNSKILLED = "unskilled"


@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
class SafeTestVersion1(PromptResponseTest):
    """Version 1.0 of SAFE test. This version is still a work in progress and subject to change.

    Assumptions about working group deliverables:
    - Prompt files:
        - Filename format:
            safe-[hazard uid]-[timestamp]-[supplier code].csv
        - CSV files include columns:
            Prompt UID, Text, Hazard UID, Locale, Persona
        - There many be multiple personas and locales in one file.
        - There is only one hazard per file.
        - There may be multiple files associated with each hazard.
          These files differ by supplier and have no overlap in prompts.
    - One hazard per test.
    - One locale per test.
    - Tests can have multiple personas (not 100% certain about this).

    Tests will closely follow the hazard taxonomy used by Llama Guard 3 (https://huggingface.co/meta-llama/Llama-Guard-3-8B#hazard-taxonomy-and-policy).
    Test UIDs follow the form "safe-{ hazard acronym }-1.0" (e.g. safe-vcr-1.0). The hazard acronyms are as follows:
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
    - sxc_prn: Sexual Content; pornographic
    """

    hazards = [
        "vcr",
        "ncr",
        "src",
        "cse",
        "dfm",
        "spc",
        "prv",
        "ipv",
        "iwp",
        "hte",
        "ssh",
        "sxc",
    ]

    def __init__(
        self,
        uid: str,
        hazard: str,
        locale: Locale,
        persona_types: List[SafePersonasVersion1],
        annotator_class: None,
        **kwargs,
    ):
        super().__init__(uid)
        assert hazard in self.hazards, f"Invalid hazard type {hazard}."
        self.hazard = hazard
        self.locale = locale
        assert len(persona_types), "At least one persona type must be specified."
        assert len(set(persona_types)) == len(
            persona_types
        ), f"Must specify a unique set of persona types, but got {persona_types}"
        self.persona_types = persona_types
        self.annotators = annotator_class(secrets=kwargs)  # type: ignore

    def _get_prompt_set_names(self):
        # TODO: Get all filenames with this test's hazard from modellab.
        return [f"safetest_{self.hazard}_09122024"]

    def get_dependencies(self) -> Mapping[str, ExternalData]:
        modellab_base_download_url = (
            "https://ai-safety-dev.uc.r.appspot.com/files/download"
        )
        dependencies = {}
        for prompt_set in self._get_prompt_set_names():
            # TODO: Ask modellab for a stable version.
            dependencies[prompt_set] = WebData(
                source_url=f"{modellab_base_download_url}/{prompt_set}.csv"
            )
        return dependencies

    def get_annotators(self) -> Mapping[str, Annotator]:
        return self.annotators.annotators

    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        sut_options = SUTOptions(
            max_tokens=500,
            temperature=0.01,
        )
        test_items: List[TestItem] = []
        for prompt_set in self._get_prompt_set_names():
            data_file = dependency_helper.get_local_path(prompt_set)
            with open(data_file, "r") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    text = row["prompt_text"].strip()
                    if not text:
                        continue

                    # Check that prompt is for correct hazard/persona/locale.
                    file_hazard = row["hazard"]
                    persona = SafePersonasVersion1(row["persona"])
                    locale = Locale(row["locale"])
                    assert (
                        file_hazard == self.hazard
                    ), f"Expected {data_file} to have {self.hazard}, but had {file_hazard}."
                    if persona not in self.persona_types:
                        continue
                    if locale != self.locale:
                        continue

                    prompt = PromptWithContext(
                        prompt=TextPrompt(text=text, options=sut_options),
                        source_id=row["prompt_uid"],
                    )
                    test_items.append(
                        TestItem(
                            prompts=[prompt],
                            context=SafeTestItemContext(persona_type=persona),
                        ),
                    )
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


ALL_PERSONAS = [
    SafePersonasVersion1.NORMAL,
    SafePersonasVersion1.SKILLED,
    SafePersonasVersion1.UNSKILLED,
]

# TODO: Remove version number from 1.0 UIDs.
#  Will need to consider how to handle hazard acronyms that are shared between v0.5 and v1.0 (e.g. safe-vcr).
#  This will likely involve some sort of versioning mechanism in the test registry.


# default llama guard annotator, always
injectable_secrets = {
    k: InjectSecret(v) for k, v in DefaultAnnotatorSet.required_secret_types.items()
}
for hazard in SafeTestVersion1.hazards:
    TESTS.register(
        SafeTestVersion1,
        f"safe-{hazard}-1.0",
        hazard,
        Locale.EN_US,
        ALL_PERSONAS,
        annotator_class=DefaultAnnotatorSet,
        **injectable_secrets,
    )

# private annotators, if available
try:
    from modelgauge.private_ensemble_annotator_set import EnsembleAnnotatorSet

    register_private = True
except:
    register_private = False

if register_private:
    injectable_secrets = {
        k: InjectSecret(v)  # type: ignore
        for k, v in EnsembleAnnotatorSet.required_secret_types.items()
    }
    for hazard in SafeTestVersion1.hazards:
        TESTS.register(
            SafeTestVersion1,
            f"safe-{hazard}-1.0-private",
            hazard,
            Locale.EN_US,
            ALL_PERSONAS,
            annotator_class=EnsembleAnnotatorSet,
            **injectable_secrets,
        )
