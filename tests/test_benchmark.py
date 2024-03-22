import pathlib
from typing import List

import pytest

from coffee.benchmark import (
    BiasHazardDefinition,
    GeneralChatBotBenchmarkDefinition,
    HazardScore,
    STANDARDS,
    ToxicityHazardDefinition,
    ValueEstimate,
)
from coffee.newhelm_runner import NewhelmSut

SIMPLE_BBQ_DATA = pathlib.Path(__file__).parent / "data/newhelm_runs/bbq"
SIMPLE_TOXICITY_DATA = pathlib.Path(__file__).parent / "data/newhelm_runs/toxicity"


def test_benchmark_definition_basics():
    mbb = GeneralChatBotBenchmarkDefinition()
    assert mbb.name() == "General Chat Bot"
    assert mbb.path_name() == "general_chat_bot_benchmark"
    h = mbb.hazards()
    assert len(h) == 2
    assert h[0].__class__ == BiasHazardDefinition
    assert h[1].__class__ == ToxicityHazardDefinition


def test_hazard_definition_basics(fake_secrets):
    import newhelm.tests.bbq

    h = BiasHazardDefinition()
    assert h.name() == "Bias"
    assert h.reference_standard() == 0.29454545454545455
    tests = h.tests(secrets=fake_secrets)
    assert len(tests) == 11
    for t in tests:
        assert t.__class__ == newhelm.tests.bbq.BBQ


def test_hazard_score_basics(start_time, end_time):
    bd = BiasHazardDefinition()
    a_score = bd.reference_standard()
    hs = HazardScore(hazard_definition=bd, score=ValueEstimate.make(a_score, 50), test_scores={})
    assert hs.hazard_definition == bd
    assert hs.score.estimate == a_score
    assert hs.stars() == 3.0


def test_hazard_score_with_max_standard(start_time, end_time):
    def dummy(raw_score):
        hd = BiasHazardDefinition()
        hd._three_star_standard = 1
        return HazardScore(hd, raw_score, start_time, end_time)


def test_newhelm_sut_display_name_and_name():
    assert NewhelmSut.GPT2.display_name == "OpenAI GPT-2"
    assert NewhelmSut.GPT2.name == "GPT2"
    assert NewhelmSut.LLAMA_2_7B.display_name == "Meta Llama 2, 7b parameters"
    assert NewhelmSut.LLAMA_2_7B.name == "LLAMA_2_7B"


def test_standards():
    assert STANDARDS.reference_standard_for("Bias")


def test_value_estimate_make():
    assert ValueEstimate.make(0.5, 10) == ValueEstimate(
        lower=0.1870860284474045,
        estimate=0.5,
        upper=0.8129139715525955,
        samples=10,
    )


def test_value_estimate_combine():
    estimates = [
        ValueEstimate(lower=0.1, estimate=0.6, upper=0.7, samples=10),
        ValueEstimate(lower=0.2, estimate=0.3, upper=0.9, samples=20),
    ]
    assert ValueEstimate.combine(estimates) == ValueEstimate(
        lower=0.2265576488285767,
        estimate=0.4,  # Weighted average of 0.6 and 0.3.
        upper=0.5939650699481813,
        samples=30,
    )


def test_value_estimate_combine_precise_small_samples():
    estimates = [
        ValueEstimate(lower=0.1, estimate=0.1111111, upper=0.7, samples=2),
        ValueEstimate(lower=0.2, estimate=0.3333333, upper=0.9, samples=2),
    ]
    result = ValueEstimate.combine(estimates)
    assert result.estimate == pytest.approx(0.222222)


def test_value_estimate_scaling_up():
    estimates: List[ValueEstimate] = []
    for samples in range(5, 100):
        estimates.append(ValueEstimate.make(0.35, samples))
    for i, estimate in enumerate(estimates):
        assert estimate.estimate == pytest.approx(0.35)
        assert 0 <= estimate.lower <= estimate.estimate
        assert estimate.estimate <= estimate.upper <= 1
        if i > 0:
            # Assert that the range gets smaller as samples increase.
            estimate_range = estimate.upper - estimate.lower
            previous_range = estimates[i - 1].upper - estimates[i - 1].lower
            assert estimate_range < previous_range, f"{estimate} vs {estimates[i-1]}"


def test_value_estimate_min():
    estimates = [
        ValueEstimate(lower=0.1, estimate=0.5, upper=0.7, samples=10),
        ValueEstimate(lower=0.2, estimate=0.3, upper=0.9, samples=20),
    ]
    minimum = ValueEstimate.min(estimates)
    assert minimum.estimate == 0.3
