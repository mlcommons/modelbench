from modelgauge.ensemble_annotator_set import EnsembleAnnotatorSet, EnsembleStrategy
from modelgauge.safety_model_response import SafetyModelResponse


class FakeEnsemble(EnsembleAnnotatorSet):

    def evaluate(self, item):
        return {"ensemble_vote": 1.0}


class FakeEnsembleStrategy(EnsembleStrategy):
    def compute_response(self, annotations):
        return SafetyModelResponse(
            is_safe=True,
            safety_categories=[],
            is_valid=True,
        )


class BadEnsembleStrategy(FakeEnsembleStrategy):
    def compute_response(self, annotations):
        raise RuntimeError("Failed to compute response")
