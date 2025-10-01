from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.ensemble_annotator import EnsembleAnnotator

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
    import modelgauge.ensemble.majority_vote_ensemble_strategy  # type: ignore
    from modelgauge.safety_model_response import SafetyModelResponse  # type: ignore
except:
    raise NotImplementedError("Private annotators are not available.")

# Safety annotator ids
SAFETY_ANNOTATOR_V1_0_UID = "safety_v1.0"
SAFETY_ANNOTATOR_V1_1_UID = "safety_v1.1"
SAFETY_ANNOTATOR_COMBINER_STRATEGY = "majority_vote"

# Security annotator ids
SECURITY_ANNOTATOR_V0_5 = GEMMA_3x27B_CONFIG.name


ANNOTATORS.register(
    EnsembleAnnotator,
    SAFETY_ANNOTATOR_V1_0_UID,
    annotators=[
        MISTRAL_7B_RUBY_CONFIG.name,
        MISTRAL_8x22B_CONFIG.name,
        LLAMA_3_70B_CONFIG.name,
        LG3_LORA_CONFIG.name,
    ],
    strategy=SAFETY_ANNOTATOR_COMBINER_STRATEGY,
)

ANNOTATORS.register(
    EnsembleAnnotator,
    SAFETY_ANNOTATOR_V1_1_UID,
    annotators=[
        MISTRAL_7B_RUBY_CONFIG.name,
        LG3_LORA_CONFIG.name,
        GEMMA_3x27B_CONFIG.name,
    ],
    strategy=SAFETY_ANNOTATOR_COMBINER_STRATEGY,
)
