import os
from typing import Dict, Any

from modelgauge.annotator import Annotator
from modelgauge.annotator_set import AnnotatorSet
from modelgauge.config import load_secrets_from_config
from modelgauge.secret_values import InjectSecret, RequiredSecret, SecretDescription
from modelgauge.single_turn_prompt_response import TestItemAnnotations
from modelgauge.suts.together_client import TogetherApiKey

try:
    from modelgauge.annotators.ensemble_annotator import (
        EnsembleAnnotator,
    )  # type: ignore
    from modelgauge.annotators.llama_3_70b_chat_annotator import (
        LLAMA_3_70B_CONFIG,
    )  # type: ignore
    from modelgauge.annotators.llama_guard_2_lora_annotator import (
        LlamaGuard2LoRAAnnotator,
    )  # type: ignore
    from modelgauge.annotators.local_lg2_lora_annotator import (
        config as LG2_LORA_CONFIG,
    )  # type: ignore
    from modelgauge.annotators.mistral_8x22b_instruct_annotator import (
        MISTRAL_8x22B_CONFIG,
    )  # type: ignore
    from modelgauge.annotators.prompt_engineered_annotator import (
        PromptEngineeredAnnotator,
    )  # type: ignore
    from modelgauge.annotators.wildguard_annotator import (  # type: ignore
        WILDGUARD_ANNOTATOR_CONFIG,
        WildguardAnnotator,
    )
    from modelgauge.safety_model_response import SafetyModelResponse  # type: ignore
except:
    raise NotImplementedError("Private annotators are not available.")


# TODO: move this out into private repo
class VllmApiKey(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="vllm",
            key="api_key",
            instructions="Contact MLCommons admin for access.",
        )


# TODO: move this out into private repo
class HuggingFaceKey(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="huggingface",
            key="api_key",
            instructions="Add your HuggingFace token to secrets, or contact MLCommons admin.",
        )


class EnsembleAnnotatorSet(AnnotatorSet):

    annotators: dict[str, Annotator] = {}
    secrets: dict[str, Any] = {}

    required_secret_types = {
        "together_api_key": TogetherApiKey,
        "huggingface_key": HuggingFaceKey,
        "vllm_api_key": VllmApiKey,
    }

    def __init__(self, secrets):
        self.secrets = secrets
        self.__configure_vllm_annotators()
        self.__configure_huggingface_annotators()
        self.__configure_together_annotators()
        self._load_annotators()

    def _load_annotators(self):
        self.annotators = {
            MISTRAL_8x22B_CONFIG.name: PromptEngineeredAnnotator(
                uid=MISTRAL_8x22B_CONFIG.name, config=MISTRAL_8x22B_CONFIG
            ),
            LLAMA_3_70B_CONFIG.name: PromptEngineeredAnnotator(
                uid=LLAMA_3_70B_CONFIG.name, config=LLAMA_3_70B_CONFIG
            ),
            LG2_LORA_CONFIG.name: LlamaGuard2LoRAAnnotator(
                uid=LG2_LORA_CONFIG.name, config=LG2_LORA_CONFIG
            ),
            WILDGUARD_ANNOTATOR_CONFIG.name: WildguardAnnotator(
                uid=WILDGUARD_ANNOTATOR_CONFIG.name,
                config=WILDGUARD_ANNOTATOR_CONFIG,
            ),
        }

    def __configure_vllm_annotators(self):
        self.secrets["vllm_endpoint_url"] = os.environ.get("VLLM_ENDPOINT_URL", "")
        assert self.secrets[
            "vllm_endpoint_url"
        ], "Environment variable `VLLM_ENDPOINT_URL` is not set."
        LG2_LORA_CONFIG.api_key = self.secrets["vllm_api_key"].value
        LG2_LORA_CONFIG.base_url = self.secrets["vllm_endpoint_url"]

    def __configure_huggingface_annotators(self):
        WILDGUARD_ANNOTATOR_CONFIG.api_key = self.secrets["huggingface_key"]
        assert (
            WILDGUARD_ANNOTATOR_CONFIG.is_valid()
        ), "HuggingFace configuration is missing a token or endpoint URL."

    def __configure_together_annotators(self):
        MISTRAL_8x22B_CONFIG.llm_config.api_key = self.secrets["together_api_key"]
        LLAMA_3_70B_CONFIG.llm_config.api_key = self.secrets["together_api_key"]

    def evaluate(self, item: TestItemAnnotations) -> Dict[str, float]:
        annotated_completion = item.interactions[0].response.completions[0]
        individual_annotations = [
            annotated_completion.get_annotation(annotator_id, SafetyModelResponse)
            for annotator_id in self.annotators.keys()
        ]
        annotation = EnsembleAnnotator.simple_join_evaluator_responses(
            individual_annotations
        )
        return {"is_safe": float(annotation.is_safe)}
