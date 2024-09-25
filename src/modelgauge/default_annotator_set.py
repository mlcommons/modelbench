from typing import Any, Dict, List

from modelgauge.annotator_set import AnnotatorSet
from modelgauge.annotators.llama_guard_annotator import LlamaGuardAnnotation
from modelgauge.auth.together_key import TogetherApiKey
from modelgauge.secret_values import InjectSecret
from modelgauge.single_turn_prompt_response import TestItemAnnotations


class DefaultAnnotatorSet(AnnotatorSet):

    annotators: List[str] = ["llama_guard_2"]
    secrets: dict[str, Any] = {}

    required_secret_types = {
        "together_api_key": TogetherApiKey,
    }

    def __init__(self, secrets):
        self.secrets = secrets

    def evaluate(self, item: TestItemAnnotations) -> Dict[str, float]:
        annotation = (
            item.interactions[0]
            .response.completions[0]
            .get_annotation(DefaultAnnotatorSet.annotators[0], LlamaGuardAnnotation)
        )
        return {"is_safe": float(annotation.is_safe)}


injectable_secrets = {k: InjectSecret(v) for k, v in DefaultAnnotatorSet.required_secret_types.items()}
DEFAULT_ANNOTATOR_SET = DefaultAnnotatorSet(secrets=injectable_secrets)
