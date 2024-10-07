import os
from typing import Any, Dict, List

from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.annotator_set import AnnotatorSet
from modelgauge.auth.huggingface_inference_token import HuggingFaceInferenceToken
from modelgauge.auth.together_key import TogetherApiKey
from modelgauge.auth.vllm_key import VllmApiKey
from modelgauge.secret_values import InjectSecret
from modelgauge.single_turn_prompt_response import TestItemAnnotations

try:
    from modelgauge.annotators.ensemble_annotator import (
        EnsembleAnnotator,
    )  # type: ignore
    from modelgauge.annotators.llama_3_70b_chat_annotator import (
        LLAMA_3_70B_PE_TAMALPAIS_2024_09_09_CONFIG as LLAMA_3_70B_CONFIG,
    )  # type: ignore
    from modelgauge.annotators.llama_guard_2_lora_annotator import (
        LlamaGuard2LoRAAnnotator,
    )  # type: ignore
    from modelgauge.annotators.local_lg2_lora_annotator import (
        config as LG2_LORA_CONFIG,
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
        self.__configure_vllm_annotators()
        self.__configure_huggingface_annotators()
        self.__configure_together_annotators()
        self._register_annotators()

    def _register_annotators(self):
        # TODO: Register annotators in secret repo.
        ANNOTATORS.register(
            PromptEngineeredAnnotator(uid=MISTRAL_8x22B_CONFIG.name, config=MISTRAL_8x22B_CONFIG),
        )
        ANNOTATORS.register(PromptEngineeredAnnotator(uid=LLAMA_3_70B_CONFIG.name, config=LLAMA_3_70B_CONFIG))
        ANNOTATORS.register(LlamaGuard2LoRAAnnotator(uid=LG2_LORA_CONFIG.name, config=LG2_LORA_CONFIG))
        ANNOTATORS.register(WildguardAnnotator(uid=WILDGUARD_ANNOTATOR_CONFIG.name, config=WILDGUARD_ANNOTATOR_CONFIG))

        self.annotators = [
            MISTRAL_8x22B_CONFIG.name,
            LLAMA_3_70B_CONFIG.name,
            LG2_LORA_CONFIG.name,
            WILDGUARD_ANNOTATOR_CONFIG.name,
        ]

    def __configure_vllm_annotators(self):
        self.secrets["vllm_endpoint_url"] = os.environ.get("VLLM_ENDPOINT_URL", "")
        assert self.secrets["vllm_endpoint_url"], "Environment variable `VLLM_ENDPOINT_URL` is not set."
        LG2_LORA_CONFIG.api_key = self.secrets["vllm_api_key"].value
        LG2_LORA_CONFIG.base_url = self.secrets["vllm_endpoint_url"]

    def __configure_huggingface_annotators(self):
        WILDGUARD_ANNOTATOR_CONFIG.token = self.secrets["huggingface_inference_token"]
        assert WILDGUARD_ANNOTATOR_CONFIG.is_valid(), "HuggingFace configuration is missing a token or endpoint URL."

    def __configure_together_annotators(self):
        MISTRAL_8x22B_CONFIG.llm_config.api_key = self.secrets["together_api_key"]
        LLAMA_3_70B_CONFIG.llm_config.api_key = self.secrets["together_api_key"]

    def evaluate(self, item: TestItemAnnotations) -> Dict[str, float]:
        annotated_completion = item.interactions[0].response.completions[0]
        individual_annotations = [
            annotated_completion.get_annotation(annotator_id, SafetyModelResponse) for annotator_id in self.annotators
        ]
        annotation = EnsembleAnnotator.simple_join_evaluator_responses(individual_annotations)
        return {"is_safe": float(annotation.is_safe)}


injectable_secrets = {k: InjectSecret(v) for k, v in EnsembleAnnotatorSet.required_secret_types.items()}  # type: ignore
PRIVATE_ANNOTATOR_SET = EnsembleAnnotatorSet(secrets=injectable_secrets)
