from abc import abstractmethod
from typing import Iterable, Sequence, Tuple

import scipy
from pydantic import BaseModel


class LetterGradeMixin:
    grades = {
        1: "F",
        2: "D",
        3: "C",
        4: "B",
        5: "A",
    }

    def text_grade(self) -> str:
        return self.grades[self.numeric_grade()]

    @abstractmethod
    def numeric_grade(self):
        pass


class ValueEstimate(BaseModel):
    lower: float
    estimate: float
    upper: float
    samples: int

    @staticmethod
    def make(probability: float, samples: int) -> "ValueEstimate":
        return ValueEstimate._estimate_confidence_intervals([(probability, samples)])

    @staticmethod
    def combine(estimates: Iterable["ValueEstimate"]) -> "ValueEstimate":
        _estimates = [(v.estimate, v.samples) for v in estimates]
        return ValueEstimate._estimate_confidence_intervals(_estimates)

    @staticmethod
    def _estimate_confidence_intervals(estimates: Sequence[Tuple[float, int]]) -> "ValueEstimate":
        assert len(estimates) > 0, "Must have at least one estimate."
        successes = 0.0
        trials = 0
        for probability, samples in estimates:
            assert 0 <= probability <= 1, "Expected all estimates to be probabilities."
            assert samples > 0, "Must have a positive number of samples."
            successes += probability * samples
            trials += samples
        estimate = successes / trials

        success_count = int(round(successes))  # binomtest takes integers.
        result = scipy.stats.binomtest(success_count, trials)
        ci = result.proportion_ci()
        # Since binomtest uses an integer number of successes, it could produce
        # bounds that violate our expectations. So use "min" and "max" to protect
        # against that.
        lower = min(ci.low, estimate)
        upper = max(ci.high, estimate)
        return ValueEstimate(lower=lower, estimate=estimate, upper=upper, samples=trials)
