import pathlib
import pickle

import pytest

from coffee.benchmark import (
    GeneralChatBotBenchmarkDefinition,
    Hazard,
    STANDARDS,
)
from coffee.newhelm_runner import NewhelmSut
from newhelm.tests.real_toxicity_prompts import RealToxicityPrompts

SIMPLE_BBQ_DATA = pathlib.Path(__file__).parent / "data/newhelm_runs/bbq"
SIMPLE_TOXICITY_DATA = pathlib.Path(__file__).parent / "data/newhelm_runs/toxicity"
SIMPLE_CHAT_DATA = pathlib.Path(__file__).parent / "data/newhelm_runs/chat"


def test_benchmark_definition_basics(fake_secrets):
    mbb = GeneralChatBotBenchmarkDefinition()
    assert mbb.name() == "General Chat Bot"
    assert mbb.path_name() == "general_chat_bot_benchmark"
    tests = mbb.get_tests(secrets=fake_secrets)
    assert len(tests) == 12
    assert tests["real_toxicity_prompts"].__class__ == RealToxicityPrompts


# TODO Reintroduce Standards tests


def test_newhelm_sut_display_name_and_name():
    assert NewhelmSut.GPT2.display_name == "OpenAI GPT-2"
    assert NewhelmSut.GPT2.name == "GPT2"
    assert NewhelmSut.LLAMA_2_7B.display_name == "Meta Llama 2, 7b parameters"
    assert NewhelmSut.LLAMA_2_7B.name == "LLAMA_2_7B"


@pytest.mark.datafiles(SIMPLE_CHAT_DATA)
def test_hazard_scoring(datafiles, start_time, end_time):
    with open(pathlib.Path(datafiles) / "test_records.pickle", "rb") as out:
        helm_scores = pickle.load(out)

    mbb = GeneralChatBotBenchmarkDefinition()
    hazard_scores = mbb.score_hazards(helm_scores)
    estimates = {hs.hazard: hs.score.estimate for hs in hazard_scores}
    assert pytest.approx(estimates[Hazard.BIAS]) == 0.381818
    assert pytest.approx(estimates[Hazard.TOXICITY]) == 0.5636363


def test_standards():
    assert STANDARDS.three_star_standard_for("Bias")
