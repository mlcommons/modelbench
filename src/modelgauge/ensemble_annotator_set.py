from abc import ABC, abstractmethod
from typing import Dict, List

from modelgauge.annotator_set import AnnotatorSet
from modelgauge.safety_model_response import EnsembleSafetyModelResponse, SafetyModelResponse
from modelgauge.single_turn_prompt_response import SUTResponseAnnotations


class EnsembleStrategy(ABC):

    @abstractmethod
    def compute_response(self, annotations: Dict[str, SafetyModelResponse]) -> EnsembleSafetyModelResponse:
        pass


# Dictionary to hold registered ensemble strategies
ENSEMBLE_STRATEGIES: Dict[str, EnsembleStrategy] = {}


class EnsembleAnnotatorSet(AnnotatorSet):
    annotators: List[str] = []

    def __init__(self, annotators: List[str], strategy: EnsembleStrategy):
        self.strategy = strategy
        self.annotators = annotators

    def evaluate(self, item: SUTResponseAnnotations) -> Dict[str, float]:
        individual_annotations = {}
        for annotator_id in self.annotators:
            if annotator_id in item.annotations:
                individual_annotations[annotator_id] = item.get_annotation(annotator_id, SafetyModelResponse)
        annotation = self.strategy.compute_response(individual_annotations)
        return {"is_safe": float(annotation.is_safe)}
