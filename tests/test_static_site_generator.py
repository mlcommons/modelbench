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
from coffee.static_site_generator import StaticSiteGenerator, display_stars


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
        "static/style.css",
        "benchmarks.html",
        "GPT2_general_chat_bot_benchmark_report.html",
        "index.html",
    ],
)
@pytest.mark.datafiles(SIMPLE_BBQ_DATA)
def test_creates_files(benchmark_score, tmp_path, path):
    generator = StaticSiteGenerator()
    generator.generate([benchmark_score], tmp_path)
    assert (tmp_path / path).exists()


@pytest.mark.datafiles(SIMPLE_BBQ_DATA)
@pytest.mark.parametrize(
    "score, size, expected",
    [
        (2.0, "md", (2, 0, 3)),
        (2.0, "lg", (2, 0, 3)),
        (2.50, "md", (2, 1, 2)),
        (2.51, "md", (2, 1, 2)),
        (4.2, "lg", (4, 0, 1)),
    ],
)
def test_displays_correct_stars(score, size, expected):
    stars = display_stars(score, size)
    half_stars = stars.count("bi-star-half")
    empty_stars = stars.count("#C8CFDD")
    full_stars = stars.count("bi-star-fill") - empty_stars

    # should generate 5 total stars of correct size
    assert stars.count(f"star-span-{size}") == 5

    assert full_stars == expected[0]
    assert half_stars == expected[1]
    assert empty_stars == expected[2]
