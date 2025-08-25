from modelgauge.annotator_set import AnnotatorSet
from modelgauge.ensemble_annotator_set import EnsembleAnnotatorSet
from modelgauge.single_turn_prompt_response import SUTResponseAnnotations

try:
    from modelgauge.annotators.gemma_3_27b_annotator import GEMMA_3x27B_CONFIG
    from modelgauge.annotators.llama_3_70b_chat_annotator import (
        LLAMA_3_70B_PE_TAMALPAIS_2024_09_09_CONFIG as LLAMA_3_70B_CONFIG,
    )  # type: ignore
    from modelgauge.annotators.template_lg3_lora_annotator import (
        config as LG3_LORA_CONFIG,
    )  # type: ignore
    from modelgauge.annotators.mistral_7b_ruby_annotator import (
        config as MISTRAL_7B_RUBY_CONFIG,
    )  # type: ignore
    from modelgauge.annotators.mistral_8x22b_instruct_annotator import (
        MISTRAL_8x22B_PE_TAMALPAIS_2024_09_09_ON_TOGETHER_DEDICATED as MISTRAL_8x22B_CONFIG,
    )  # type: ignore
    from modelgauge.ensemble.majority_vote_ensemble_strategy import MajorityVoteEnsembleStrategy  # type: ignore
    from modelgauge.safety_model_response import SafetyModelResponse  # type: ignore
except:
    raise NotImplementedError("Private annotators are not available.")


PRIVATE_ANNOTATOR_SET = EnsembleAnnotatorSet(
    annotators=[
        MISTRAL_7B_RUBY_CONFIG.name,
        MISTRAL_8x22B_CONFIG.name,
        LLAMA_3_70B_CONFIG.name,
        LG3_LORA_CONFIG.name,
    ],
    strategy=MajorityVoteEnsembleStrategy(),
)


class PrivateSecurityAnnotatorSet(AnnotatorSet):
    annotators: list[str] = [GEMMA_3x27B_CONFIG.name]

    def evaluate(self, item: SUTResponseAnnotations) -> dict[str, float]:
        # Can't use MajorityVoteEnsembleStrategy here because we only have one annotator.
        annotation = item.get_annotation(GEMMA_3x27B_CONFIG.name, SafetyModelResponse)
        if not annotation.is_valid:
            is_safe = 0.0
        else:
            is_safe = float(annotation.is_safe)
        return {"is_safe": is_safe}


PRIVATE_SECURITY_ANNOTATOR_SET = PrivateSecurityAnnotatorSet()
