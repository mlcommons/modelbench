import csv
import os
from enum import Enum
from typing import Dict, List, Mapping, Optional

from modelgauge.aggregations import get_measurement_stats_by_key
from modelgauge.annotator import Annotator
from modelgauge.annotators.llama_guard_annotator import LlamaGuardAnnotation, LlamaGuardAnnotator, LlamaGuardVersion
from modelgauge.auth.huggingface_inference_token import HuggingFaceInferenceToken
from modelgauge.auth.together_key import TogetherApiKey
from modelgauge.auth.vllm_key import VllmApiKey
from modelgauge.base_test import PromptResponseTest
from modelgauge.dependency_helper import DependencyHelper
from modelgauge.external_data import ExternalData, WebData
from modelgauge.prompt import SUTOptions, TextPrompt
from modelgauge.secret_values import InjectSecret
from modelgauge.single_turn_prompt_response import MeasuredTestItem, PromptWithContext, TestItem, TestItemAnnotations
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.test_decorator import modelgauge_test
from modelgauge.test_registry import TESTS
from modelgauge.tests.safe import PersonaResult, SafeTestItemContext, SafeTestResult

try:
    from modelgauge.annotators.ensemble_annotator import EnsembleAnnotator  # type: ignore
    from modelgauge.annotators.llama_3_70b_chat_annotator import LLAMA_3_70B_CONFIG  # type: ignore
    from modelgauge.annotators.llama_guard_2_lora_annotator import LlamaGuard2LoRAAnnotator  # type: ignore
    from modelgauge.annotators.llama_guard_2_lora_annotator import LlamaGuard2LoRAConfig  # type: ignore
    from modelgauge.annotators.local_lg2_lora_annotator import config as LG2_LORA_CONFIG  # type: ignore
    from modelgauge.annotators.mistral_8x22b_instruct_annotator import (
        MISTRAL_8x22B_CONFIG,
    )  # type: ignore
    from modelgauge.annotators.prompt_engineered_annotator import (
        PromptEngineeredAnnotator,
    )  # type: ignore
    from modelgauge.annotators.wildguard_annotator import WILDGUARD_ANNOTATOR_CONFIG  # type: ignore
    from modelgauge.annotators.wildguard_annotator import WildguardAnnotator  # type: ignore
    from modelgauge.safety_model_response import SafetyModelResponse  # type: ignore

    PRIVATE_ANNOTATORS_AVAILABLE = True
except ImportError:
    PRIVATE_ANNOTATORS_AVAILABLE = False


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
        together_api_key: TogetherApiKey,
        vllm_api_key: Optional[VllmApiKey] = None,
        huggingface_inference_token: Optional[HuggingFaceInferenceToken] = None,
        use_private_annotators: bool = False,
    ):
        """Uses llamaguard 2 if use_private_annotators is False."""
        super().__init__(uid)
        assert hazard in self.hazards, f"Invalid hazard type {hazard}."
        self.hazard = hazard
        self.locale = locale
        assert len(persona_types), "At least one persona type must be specified."
        assert len(set(persona_types)) == len(
            persona_types
        ), f"Must specify a unique set of persona types, but got {persona_types}"
        self.persona_types = persona_types
        self.use_private_annotators = use_private_annotators

        # Secrets
        # Any secret values that may have been pulled from the environment
        # will be replaced by secrets loaded from a secret config.
        self.together_api_key = together_api_key
        if self.use_private_annotators:
            self._configure_vllm_annotators(vllm_api_key)  # type: ignore
            self._configure_huggingface_annotators(huggingface_inference_token)  # type: ignore
            self._configure_together_annotators(together_api_key)

    # TODO: Encapsulate multipart secrets (e.g. key and URL).
    # I'm not convinced the SafeTestV1 class needs to concern itself
    # with the internals of vllm access.
    def _configure_vllm_annotators(self, vllm_api_key: VllmApiKey):
        self.vllm_api_key = vllm_api_key
        self.vllm_endpoint_url = os.environ.get("VLLM_ENDPOINT_URL")
        assert self.vllm_api_key, "VLLM API key is required for private annotators."
        assert self.vllm_endpoint_url, "Environment variable `VLLM_ENDPOINT_URL` is not set."
        LG2_LORA_CONFIG.api_key = self.vllm_api_key.value
        LG2_LORA_CONFIG.base_url = self.vllm_endpoint_url

    def _configure_huggingface_annotators(self, huggingface_inference_token: HuggingFaceInferenceToken):
        WILDGUARD_ANNOTATOR_CONFIG.token = huggingface_inference_token.value
        assert WILDGUARD_ANNOTATOR_CONFIG.is_valid(), "HuggingFace configuration is missing a token or endpoint URL."

    def _configure_together_annotators(self, together_api_key: TogetherApiKey):
        MISTRAL_8x22B_CONFIG.llm_config.api_key = together_api_key
        LLAMA_3_70B_CONFIG.llm_config.api_key = together_api_key

    def _get_prompt_set_names(self):
        # TODO: Get all filenames with this test's hazard from modellab.
        return [f"safetest_{self.hazard}_09122024"]

    def get_dependencies(self) -> Mapping[str, ExternalData]:
        modellab_base_download_url = "https://ai-safety-dev.uc.r.appspot.com/files/download"
        dependencies = {}
        for prompt_set in self._get_prompt_set_names():
            # TODO: Ask modellab for a stable version.
            dependencies[prompt_set] = WebData(source_url=f"{modellab_base_download_url}/{prompt_set}.csv")
        return dependencies

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

    def get_annotators(self) -> Mapping[str, Annotator]:
        if self.use_private_annotators:
            return self._get_private_annotators()
        return self._get_default_annotators()

    def _get_default_annotators(self):
        return {
            "llama_guard_2": LlamaGuardAnnotator(
                "llama_guard_2",
                self.together_api_key,
                llama_guard_version=LlamaGuardVersion.VERSION_2,
            )
        }

    def _get_private_annotators(self):
        return {
            MISTRAL_8x22B_CONFIG.name: PromptEngineeredAnnotator(MISTRAL_8x22B_CONFIG.name, MISTRAL_8x22B_CONFIG),
            LLAMA_3_70B_CONFIG.name: PromptEngineeredAnnotator(LLAMA_3_70B_CONFIG.name, LLAMA_3_70B_CONFIG),
            LG2_LORA_CONFIG.name: LlamaGuard2LoRAAnnotator(LG2_LORA_CONFIG.name, LG2_LORA_CONFIG),
            WILDGUARD_ANNOTATOR_CONFIG.name: WildguardAnnotator(
                WILDGUARD_ANNOTATOR_CONFIG.name,
                WILDGUARD_ANNOTATOR_CONFIG,
            ),
        }

    def measure_quality(self, item: TestItemAnnotations) -> Dict[str, float]:
        annotated_completion = item.interactions[0].response.completions[0]
        if self.use_private_annotators:
            annotator_ids = [
                MISTRAL_8x22B_CONFIG.name,
                LLAMA_3_70B_CONFIG.name,
                LG2_LORA_CONFIG.name,
                WILDGUARD_ANNOTATOR_CONFIG.name,
            ]
            individual_annotations = [
                annotated_completion.get_annotation(annotator_id, SafetyModelResponse) for annotator_id in annotator_ids
            ]

            annotation = EnsembleAnnotator.simple_join_evaluator_responses(individual_annotations)

        else:
            annotation = (
                item.interactions[0].response.completions[0].get_annotation("llama_guard_2", LlamaGuardAnnotation)
            )
        return {"is_safe": float(annotation.is_safe)}

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


TOGETHER_KEY = InjectSecret(TogetherApiKey)
ALL_PERSONAS = [
    SafePersonasVersion1.NORMAL,
    SafePersonasVersion1.SKILLED,
    SafePersonasVersion1.UNSKILLED,
]

# TODO: Remove version number from 1.0 UIDs.
#  Will need to consider how to handle hazard acronyms that are shared between v0.5 and v1.0 (e.g. safe-vcr).
#  This will likely involve some sort of versioning mechanism in the test registry.
for hazard in SafeTestVersion1.hazards:
    TESTS.register(
        SafeTestVersion1,
        f"safe-{hazard}-1.0",
        hazard,
        Locale.EN_US,
        ALL_PERSONAS,
        TOGETHER_KEY,
    )

if PRIVATE_ANNOTATORS_AVAILABLE:
    for hazard in SafeTestVersion1.hazards:
        VLLM_API_KEY = InjectSecret(VllmApiKey)
        HUGGINGFACE_INFERENCE_TOKEN = InjectSecret(HuggingFaceInferenceToken)
        TESTS.register(
            SafeTestVersion1,
            f"safe-{hazard}-1.0-private",
            hazard,
            Locale.EN_US,
            ALL_PERSONAS,
            TOGETHER_KEY,
            vllm_api_key=VLLM_API_KEY,
            huggingface_inference_token=HUGGINGFACE_INFERENCE_TOKEN,
            use_private_annotators=True,
        )
