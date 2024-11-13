import os
from typing import Any, Dict, List

from modelgauge.annotator_set import AnnotatorSet
from modelgauge.auth.huggingface_inference_token import HuggingFaceInferenceToken
from modelgauge.auth.together_key import TogetherApiKey
from modelgauge.auth.vllm_key import VllmApiKey
from modelgauge.dependency_injection import _replace_with_injected
from modelgauge.secret_values import InjectSecret
from modelgauge.single_turn_prompt_response import TestItemAnnotations

try:
    from modelgauge.annotators.llama_3_70b_chat_annotator import (
        LLAMA_3_70B_PE_TAMALPAIS_2024_09_09_CONFIG as LLAMA_3_70B_CONFIG,
    )  # type: ignore
    from modelgauge.annotators.llama_guard_3_lora_annotator import (
        LlamaGuard3LoRAAnnotator,
    )  # type: ignore
    from modelgauge.annotators.template_lg3_lora_annotator import (
        config as LG3_LORA_CONFIG,
    )  # type: ignore
    from modelgauge.annotators.mistral_8x22b_instruct_annotator import (
        MISTRAL_8x22B_PE_TAMALPAIS_2024_09_09_CONFIG as MISTRAL_8x22B_CONFIG,
    )  # type: ignore
    from modelgauge.annotators.prompt_engineered_annotator import (
        PromptEngineeredAnnotator,
    )  # type: ignore
    from modelgauge.annotators.wildguard_annotator import (  # type: ignore
        WILDGUARD_ANNOTATOR_CONFIG,
        WildguardAnnotator,
    )
    from modelgauge.ensemble.majority_vote_ensemble_strategy import MajorityVoteEnsembleStrategy  # type: ignore
    from modelgauge.safety_model_response import SafetyModelResponse  # type: ignore
except:
    raise NotImplementedError("Private annotators are not available.")


class EnsembleAnnotatorSet(AnnotatorSet):
    annotators: List[str] = []
    secrets: dict[str, Any] = {}

    required_secret_types = {
        "together_api_key": TogetherApiKey,
        "huggingface_inference_token": HuggingFaceInferenceToken,
        "vllm_api_key": VllmApiKey,
    }

    def __init__(self, secrets):
        self.secrets = secrets
        self.strategy = MajorityVoteEnsembleStrategy()
        self.__configure_vllm_annotators()
        self.__configure_huggingface_annotators()
        self.__configure_together_annotators()
        self._register_annotators()

    def _register_annotators(self):
        self.annotators = [
            MISTRAL_8x22B_CONFIG.name,
            LLAMA_3_70B_CONFIG.name,
            LG3_LORA_CONFIG.name,
            WILDGUARD_ANNOTATOR_CONFIG.name,
        ]

    def __configure_vllm_annotators(self):
        self.secrets["vllm_endpoint_url"] = os.environ.get("VLLM_ENDPOINT_URL", "")
        assert self.secrets["vllm_endpoint_url"], "Environment variable `VLLM_ENDPOINT_URL` is not set."
        LG3_LORA_CONFIG.api_key = self.secrets["vllm_api_key"].value
        LG3_LORA_CONFIG.base_url = self.secrets["vllm_endpoint_url"]

    def __configure_huggingface_annotators(self):
        WILDGUARD_ANNOTATOR_CONFIG.token = self.secrets["huggingface_inference_token"].value
        assert WILDGUARD_ANNOTATOR_CONFIG.is_valid(), "HuggingFace configuration is missing a token or endpoint URL."

    def __configure_together_annotators(self):
        MISTRAL_8x22B_CONFIG.llm_config.api_key = self.secrets["together_api_key"]
        LLAMA_3_70B_CONFIG.llm_config.api_key = self.secrets["together_api_key"]

    def evaluate(self, item: TestItemAnnotations) -> Dict[str, float]:
        annotated_completion = item.interactions[0].response.completions[0]
        individual_annotations = {}
        for annotator_id in self.annotators:
            if annotator_id in annotated_completion.annotations:
                individual_annotations[annotator_id] = annotated_completion.get_annotation(
                    annotator_id, SafetyModelResponse
                )
        annotation = self.strategy.compute_response(individual_annotations)
        return {"is_safe": float(annotation.is_safe)}


def ensemble_secrets(all_secrets):
    injectable_secrets = {
        k: InjectSecret(v) for k, v in EnsembleAnnotatorSet.required_secret_types.items()
    }  # type: ignore
    injected_secrets = {k: _replace_with_injected(v, all_secrets) for k, v in injectable_secrets.items()}
    return injected_secrets


# PRIVATE_ANNOTATOR_SET = EnsembleAnnotatorSet(secrets=ensemble_secrets(load_secrets_from_config()))
