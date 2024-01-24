import pathlib

SIMPLE_BBQ_DATA = pathlib.Path(__file__).parent / "data/full_runs/simple_bbq"

import pytest

from coffee.helm import HelmSut
from coffee.benchmark import (
    GeneralChatBotBenchmarkDefinition,
    BiasHarmDefinition,
    HarmScore,
    BenchmarkScore,
    ToxicityHarmDefinition,
)
from coffee.static_site_generator import StaticSiteGenerator


@pytest.fixture()
def benchmark_score():
    bd = GeneralChatBotBenchmarkDefinition()
    bs = BenchmarkScore(
        bd,
        HelmSut.GPT2,
        [
            HarmScore(BiasHarmDefinition(), 0.5),
            HarmScore(ToxicityHarmDefinition(), 0.8),
        ],
    )
    return bs


@pytest.mark.parametrize(
    "path",
    [
        "general_chat_bot_benchmark.html",
        "static/images/ml_commons_logo.png",
        "benchmarks.html",
    ],
)
@pytest.mark.datafiles(SIMPLE_BBQ_DATA)
def test_creates_files(benchmark_score, tmp_path, path):
    generator = StaticSiteGenerator()
    generator.generate([benchmark_score], tmp_path)
    assert (tmp_path / path).exists()


@pytest.mark.datafiles(SIMPLE_BBQ_DATA)
@pytest.mark.parametrize(
    "score,expected",
    [
        (2.0, (2, False, 3)),
        (2.49, (2, False, 3)),
        (2.50, (2, True, 2)),
        (2.51, (2, True, 2)),
        (4.0, (4, False, 1)),
    ],
)
def test_displays_correct_stars(
    benchmark_score, cwd_tmpdir, monkeypatch, score, expected
):
    monkeypatch.setattr(benchmark_score, "stars", lambda: score)
    generator = StaticSiteGenerator()
    stars = generator.calculate_stars(benchmark_score)
    assert stars == expected
