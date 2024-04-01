import pytest

from coffee.hazards import HazardScore, SafeCaeHazard
from coffee.scoring import ValueEstimate


class TestHazardScoreComparisons:
    @pytest.fixture
    def high_hazard_score(self):
        hs = HazardScore(hazard_definition=SafeCaeHazard(), score=ValueEstimate.make(1.0, 10), test_scores={})
        return hs

    @pytest.fixture
    def low_hazard_score(self):
        hs = HazardScore(hazard_definition=SafeCaeHazard(), score=ValueEstimate.make(0.1, 10), test_scores={})
        return hs

    def test_lt(self, high_hazard_score, low_hazard_score):
        assert low_hazard_score < high_hazard_score

    def test_gt(self, high_hazard_score, low_hazard_score):
        assert high_hazard_score > low_hazard_score

    def test_gte(self, high_hazard_score, low_hazard_score):
        assert high_hazard_score >= low_hazard_score
