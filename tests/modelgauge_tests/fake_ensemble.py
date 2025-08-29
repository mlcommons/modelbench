from modelgauge.ensemble_annotator_set import EnsembleAnnotatorSet, EnsembleStrategy
from modelgauge.safety_model_response import EnsembleSafetyModelResponse


class FakeEnsemble(EnsembleAnnotatorSet):

    def evaluate(self, item):
        return {"ensemble_vote": 1.0}


class FakeEnsembleStrategy(EnsembleStrategy):
    def compute_response(self, annotations):
        return {"is_safe": True, "joined_responses": annotations}


class BadEnsembleStrategy(FakeEnsembleStrategy):
    def compute_response(self, annotations):
        raise RuntimeError("Failed to compute response")


class ValidEnsembleStrategy(FakeEnsembleStrategy):
    def compute_response(self, annotations):
        return EnsembleSafetyModelResponse(
            is_safe=True,
            joined_responses=annotations,
            safety_categories=[],
        )
