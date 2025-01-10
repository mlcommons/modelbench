import os
from typing import Any, Dict, List

from modelgauge.annotator_set import AnnotatorSet
from modelgauge.auth.together_key import TogetherApiKey
from modelgauge.dependency_injection import _replace_with_injected
from modelgauge.secret_values import InjectSecret
from modelgauge.single_turn_prompt_response import TestItemAnnotations

try:
    from modelgauge.annotators.llama_3_70b_chat_annotator import (
        LLAMA_3_70B_PE_TAMALPAIS_2024_09_09_CONFIG as LLAMA_3_70B_CONFIG,
    )  # type: ignore
    from modelgauge.annotators.template_lg3_lora_annotator import (
        config as LG3_LORA_CONFIG,
        Lg3LoraVllmApiKey,
        Lg3LoraVllmEndpointUrl,
    )  # type: ignore
    from modelgauge.annotators.mistral_7b_ruby_annotator import (
        config as MISTRAL_7B_RUBY_CONFIG,
        Mistral7bVllmApiKey,
        Mistral7bVllmEndpointUrl,
    )  # type: ignore
    from modelgauge.annotators.mistral_8x22b_instruct_annotator import (
        MISTRAL_8x22B_PE_TAMALPAIS_2024_09_09_CONFIG as MISTRAL_8x22B_CONFIG,
    )  # type: ignore
    from modelgauge.ensemble.majority_vote_ensemble_strategy import MajorityVoteEnsembleStrategy  # type: ignore
    from modelgauge.safety_model_response import SafetyModelResponse  # type: ignore
except:
    raise NotImplementedError("Private annotators are not available.")


class EnsembleAnnotatorSet(AnnotatorSet):
    annotators: List[str] = []
    secrets: dict[str, Any] = {}

    required_secret_types = {
        "together_api_key": TogetherApiKey,
        "lg3_vllm_api_key": Lg3LoraVllmApiKey,
        "lg3_vllm_endpoint_url": Lg3LoraVllmEndpointUrl,
        "mistral_7b_vllm_api_key": Mistral7bVllmApiKey,
        "mistral_7b_vllm_endpoint_url": Mistral7bVllmEndpointUrl,
    }

    def __init__(self, secrets):
        self.secrets = secrets
        self.strategy = MajorityVoteEnsembleStrategy()
        self.annotators = [
            MISTRAL_7B_RUBY_CONFIG.name,
            MISTRAL_8x22B_CONFIG.name,
            LLAMA_3_70B_CONFIG.name,
            LG3_LORA_CONFIG.name,
        ]

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
