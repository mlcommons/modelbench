from modelgauge.annotation import SafetyAnnotation
from modelgauge.ensemble_strategies import EnsembleStrategy


class FakeEnsembleStrategy(EnsembleStrategy):
    def compute_response(self, annotations):
        return SafetyAnnotation(
            is_safe=True,
            is_valid=True,
        )


class BadEnsembleStrategy(FakeEnsembleStrategy):
    def compute_response(self, annotations):
        raise RuntimeError("Failed to compute response")
