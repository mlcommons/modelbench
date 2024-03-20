import pathlib
import pickle
from datetime import timedelta

import pytest

from coffee.benchmark import (
    BiasHazardDefinition,
    GeneralChatBotBenchmarkDefinition,
    BenchmarkScore,
    HazardScore,
    ToxicityHazardDefinition,
    ValueEstimate,
    quantize_stars,
    STANDARDS,
)
from coffee.newhelm_runner import NewhelmSut

SIMPLE_BBQ_DATA = pathlib.Path(__file__).parent / "data/newhelm_runs/bbq"
SIMPLE_TOXICITY_DATA = pathlib.Path(__file__).parent / "data/newhelm_runs/toxicity"


def test_benchmark_definition_basics(fake_secrets):
    mbb = GeneralChatBotBenchmarkDefinition(secrets=fake_secrets)
    assert mbb.name() == "General Chat Bot"
    assert mbb.path_name() == "general_chat_bot_benchmark"


# TODO: reintroduce standard tests


def test_newhelm_sut_display_name_and_name():
    assert NewhelmSut.GPT2.display_name == "OpenAI GPT-2"
    assert NewhelmSut.GPT2.name == "GPT2"
    assert NewhelmSut.LLAMA_2_7B.display_name == "Meta Llama 2, 7b parameters"
    assert NewhelmSut.LLAMA_2_7B.name == "LLAMA_2_7B"


def test_standards():
    assert STANDARDS.three_star_standard_for("Bias")


def test_value_estimate_make():
    values = [0.5, 1.0]
    assert ValueEstimate.make(values) == ValueEstimate(
        lower=0.5,
        estimate=0.75,
        upper=1.0,
    )


def test_value_estimate_combine():
    estimates = [
        ValueEstimate(lower=0.1, estimate=0.5, upper=0.7),
        ValueEstimate(lower=0.2, estimate=0.3, upper=0.9),
    ]
    assert ValueEstimate.combine(estimates) == ValueEstimate(
        lower=0.1,
        estimate=0.4,
        upper=0.9,
    )


def test_value_estimate_min():
    estimates = [
        ValueEstimate(lower=0.1, estimate=0.5, upper=0.7),
        ValueEstimate(lower=0.2, estimate=0.3, upper=0.9),
    ]
    minimum = ValueEstimate.min(estimates)
    assert minimum.estimate == 0.3
