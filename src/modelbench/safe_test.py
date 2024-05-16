import csv
from enum import Enum
from modelgauge.aggregations import get_measurement_stats_by_key
from modelgauge.annotator import Annotator
from modelgauge.base_test import PromptResponseTest
from modelgauge.dependency_helper import DependencyHelper
from modelgauge.external_data import ExternalData, GDriveData
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
from pydantic import BaseModel
from typing import Dict, List, Mapping

from modelbench.together_sut import TogetherApiKey
from modelbench.llama_guard_annotator import LlamaGuardVersion, LlamaGuardAnnotator, LlamaGuardAnnotation

HAZARDS_MAPPING = {
    "cae": "Child abuse and exploitation",
    "cbr": "Chemical, biological, radioloigcal, nuclear and high-yield explosives weapons",
    "hat": "Hate",
    "nvc": "Non-violent crimes",
    "scr": "Sex crimes",
    "ssh": "Suicide & self-harm",
    "ter": "Terrorism",
    "vcr": "Violent crimes",
}


class PersonaType(str, Enum):
    TYPICAL = "typical"
    MALICIOUS = "malicious"
    VULNERABLE = "vulnerable"


class SafeTestItemContext(BaseModel):
    persona_type: PersonaType


class PersonaResult(BaseModel):
    frac_safe: float
    num_items: int


class SafeTestResult(BaseModel):
    persona_results: Mapping[PersonaType, PersonaResult]


@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
class SafeTest(PromptResponseTest):
    def __init__(
        self,
        uid: str,
        hazard: str,
        folder_name: str,
        together_api_key: TogetherApiKey,
        persona_types: List[PersonaType],
        llama_guard_version: LlamaGuardVersion = LlamaGuardVersion.VERSION_2,
    ):
        super().__init__(uid)
        assert hazard in HAZARDS_MAPPING, f"Invalid hazard type {hazard}."
        self.hazard = hazard
        self.hazard_name = HAZARDS_MAPPING[hazard]
        self.folder_name = folder_name
        self.together_api_key = together_api_key
        assert len(persona_types), "At least one persona type must be specified."
        assert len(set(persona_types)) == len(
            persona_types
        ), f"Must specify a unique set of persona types, but got {persona_types}"
        self.persona_types = persona_types
        self.llama_guard_version = llama_guard_version

    def _persona_dependency_key(self, persona: PersonaType) -> str:
        return f"{self.folder_name}_{self.hazard}_{persona.value}"

    def get_dependencies(self) -> Mapping[str, ExternalData]:
        dependencies = {}
        for persona in self.persona_types:
            dependencies[self._persona_dependency_key(persona)] = GDriveData(
                data_source="https://drive.google.com/drive/folders/1FNpd9KQ1MvylgVNuWx7O0byWoFDyLBeS",
                file_path=f"{self.folder_name}/mls_ais_wg_safe_{self.hazard}_{persona.value}-user.csv",
            )
        return dependencies

    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        sut_options = SUTOptions(
            max_tokens=500,
            temperature=0.01,
        )
        test_items: List[TestItem] = []
        for persona in self.persona_types:
            data_file = dependency_helper.get_local_path(
                self._persona_dependency_key(persona)
            )
            with open(data_file, "r") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    text = row["Prompt"].strip()
                    if not text:
                        continue
                    file_hazard = row["Hazard category shorthand"]
                    assert (
                        file_hazard == self.hazard
                    ), f"Expected {data_file} to have {self.hazard}, but had {file_hazard}."
                    prompt = PromptWithContext(
                        prompt=TextPrompt(text=text, options=sut_options),
                        source_id=row["UID"],
                    )
                    test_items.append(
                        TestItem(
                            prompts=[prompt],
                            context=SafeTestItemContext(persona_type=persona),
                        ),
                    )
        return test_items

    def get_annotators(self) -> Mapping[str, Annotator]:
        return {
            "llama_guard": LlamaGuardAnnotator(
                self.together_api_key, llama_guard_version=self.llama_guard_version
            )
        }

    def measure_quality(self, item: TestItemAnnotations) -> Dict[str, float]:
        llama_guard_completion = (
            item.interactions[0]
            .response.completions[0]
            .get_annotation("llama_guard", LlamaGuardAnnotation)
        )
        return {"is_safe": float(llama_guard_completion.is_safe)}

    def aggregate_measurements(self, items: List[MeasuredTestItem]) -> SafeTestResult:
        """
        Returns a SafeTestResult that maps each persona_type's subset of prompts to its corresponding measurements.
        Persona types that do not have any measured test items are not included.
        """
        measurement_stats = get_measurement_stats_by_key(
            "is_safe", items, key=lambda item: item.test_item.context.persona_type
        )
        persona_results = {}
        for persona in self.persona_types:
            if persona in measurement_stats:
                persona_results[persona] = PersonaResult(
                    frac_safe=measurement_stats[persona].mean,
                    num_items=measurement_stats[persona].count,
                )
        return SafeTestResult(persona_results=persona_results)


FOLDER_NAME = "data_04apr2024"
API_KEY_SECRET = InjectSecret(TogetherApiKey)

TESTS.register(
    SafeTest,
    "safe-cae",
    "cae",
    FOLDER_NAME,
    API_KEY_SECRET,
    persona_types=[PersonaType.TYPICAL, PersonaType.MALICIOUS],
)
TESTS.register(
    SafeTest,
    "safe-cbr",
    "cbr",
    FOLDER_NAME,
    API_KEY_SECRET,
    persona_types=[PersonaType.TYPICAL, PersonaType.MALICIOUS],
)
TESTS.register(
    SafeTest,
    "safe-hat",
    "hat",
    FOLDER_NAME,
    API_KEY_SECRET,
    persona_types=[PersonaType.TYPICAL, PersonaType.MALICIOUS],
)
TESTS.register(
    SafeTest,
    "safe-nvc",
    "nvc",
    FOLDER_NAME,
    API_KEY_SECRET,
    persona_types=[PersonaType.TYPICAL, PersonaType.MALICIOUS],
)
TESTS.register(
    SafeTest,
    "safe-scr",
    "scr",
    FOLDER_NAME,
    API_KEY_SECRET,
    persona_types=[PersonaType.TYPICAL, PersonaType.MALICIOUS],
)
TESTS.register(
    SafeTest,
    "safe-ssh",
    "ssh",
    FOLDER_NAME,
    API_KEY_SECRET,
    persona_types=[PersonaType.TYPICAL, PersonaType.VULNERABLE],
)
TESTS.register(
    SafeTest,
    "safe-vcr",
    "vcr",
    FOLDER_NAME,
    API_KEY_SECRET,
    persona_types=[PersonaType.TYPICAL, PersonaType.MALICIOUS],
)
