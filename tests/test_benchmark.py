import pytest

from coffee.benchmark import (
    BiasHarmDefinition,
    GeneralChatBotBenchmarkDefinition,
    BenchmarkScore,
    HarmScore,
    ToxicityHarmDefinition,
    quantize_stars,
)
from coffee.helm import HelmSut, BbqHelmTest, HelmResult
from .test_helm_runner import SIMPLE_BBQ_DATA, SIMPLE_TOXICITY_DATA


def test_benchmark_definition_basics():
    mbb = GeneralChatBotBenchmarkDefinition()
    assert mbb.name() == "General Chat Bot"
    assert mbb.path_name() == "general_chat_bot_benchmark"
    h = mbb.harms()
    assert len(h) == 2
    assert h[0].__class__ == BiasHarmDefinition
    assert h[1].__class__ == ToxicityHarmDefinition


def test_harm_definition_basics():
    h = BiasHarmDefinition()
    assert h.name() == "Bias"
    t = h.tests()
    assert len(t) == 1
    assert t[0].__class__ == BbqHelmTest


def test_harm_score_basics():
    bd = BiasHarmDefinition()
    hs = HarmScore(bd, 0.6)
    assert hs.harm_definition() == bd
    assert hs.value() == 0.6
    assert hs.stars() == 3.0


def test_benchmark_score_basics():
    bd = GeneralChatBotBenchmarkDefinition()
    scores = [
        HarmScore(BiasHarmDefinition(), 0.5),
        HarmScore(ToxicityHarmDefinition(), 1.0),
    ]
    bs = BenchmarkScore(bd, HelmSut.GPT2, scores)
    assert bs.benchmark_definition == bd
    assert bs.sut == HelmSut.GPT2
    assert bs.harm_scores == scores
    assert bs.value() == 0.75
    assert bs.stars() == 4.0


@pytest.mark.datafiles(SIMPLE_BBQ_DATA)
def test_bias_scoring(datafiles):
    ht = BbqHelmTest()
    hr = HelmResult([ht], [HelmSut.GPT2], datafiles, None)
    helm_scores = hr.load_scores()
    bd = BiasHarmDefinition()
    harm_score: HarmScore = bd.score(helm_scores.for_sut(HelmSut.GPT2))
    assert pytest.approx(harm_score.value()) == 0.45
    assert harm_score.stars() == 2.0


@pytest.mark.datafiles(SIMPLE_TOXICITY_DATA)
def test_toxicity_scoring(datafiles):
    ht = ToxicityHarmDefinition().tests()[0]
    hr = HelmResult([ht], [HelmSut.GPT2], datafiles, None)
    helm_scores = hr.load_scores()
    bd = ToxicityHarmDefinition()
    harm_score: HarmScore = bd.score(helm_scores.for_sut(HelmSut.GPT2))
    assert pytest.approx(harm_score.value()) == 0.86
    assert harm_score.stars() == 4.5


def test_quantize_stars():
    assert 0 == quantize_stars(0)
    assert 5 == quantize_stars(5)

    assert 4.5 == quantize_stars(4.5)

    assert 4.5 == quantize_stars(4.26)
    assert 4.5 == quantize_stars(4.3)
    assert 4.5 == quantize_stars(4.4)
    assert 4.5 == quantize_stars(4.6)
    assert 4.5 == quantize_stars(4.7)
    assert 4.5 == quantize_stars(4.74)
