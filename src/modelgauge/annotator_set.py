from abc import ABC, abstractmethod
from typing import Dict, List


from modelgauge.annotation import EnsembleSafetyModelResponse, SafetyModelResponse, SafetyAnnotation
from modelgauge.single_turn_prompt_response import SUTResponseAnnotations


class AnnotatorSet(ABC):
    @property
    def annotators(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass


class MissingAnnotationError(Exception):
    pass


# NOTE: we should try to remove the need for AnnotatorSet entirely, but
# pushing that to future work.
class BasicAnnotatorSet(AnnotatorSet):
    annotators: List[str] = []

    def __init__(self, annotator_uid: str):
        self.annotator_uid = annotator_uid
        self.annotators = [annotator_uid]

    def evaluate(self, item: SUTResponseAnnotations) -> Dict[str, float]:
        annotation = get_safety_model_response(item, self.annotator_uid)
        if not annotation.is_valid:
            is_safe = 0.0
        else:
            is_safe = float(annotation.is_safe)
        return {"is_safe": is_safe}


def get_safety_model_response(item: SUTResponseAnnotations, annotator_uid: str) -> SafetyAnnotation:
    if annotator_uid not in item.annotations:
        raise MissingAnnotationError(f"Missing annotation for {annotator_uid}")

    annotation: SafetyModelResponse
    return item.get_annotation(annotator_uid)
