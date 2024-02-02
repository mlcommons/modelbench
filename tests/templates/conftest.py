from itertools import groupby
import pytest
import pathlib

from jinja2 import Environment, FileSystemLoader

from coffee.helm_runner import HelmSut
from coffee.benchmark import (
    GeneralChatBotBenchmarkDefinition,
    BiasHarmDefinition,
    HarmScore,
    BenchmarkScore,
    ToxicityHarmDefinition,
)
from coffee.static_site_generator import STARS_DESCRIPTION, display_stars


def _benchmark_score() -> BenchmarkScore:
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


@pytest.fixture()
def benchmark_score() -> BenchmarkScore:
    return _benchmark_score()


@pytest.fixture()
def grouped_benchmark_scores() -> dict[str, list[BenchmarkScore]]:
    benchmark_scores_dict = {}
    for benchmark_definition, grouped_benchmark_scores in groupby(
        [_benchmark_score()], lambda x: x.benchmark_definition
    ):
        grouped_benchmark_scores_list: list = list(grouped_benchmark_scores)
        benchmark_scores_dict[benchmark_definition] = grouped_benchmark_scores_list
    return benchmark_scores_dict


@pytest.fixture()
def template_env() -> Environment:
    template_dir = pathlib.Path(__file__).parent.parent.parent / "src" / "coffee" / "templates"
    env = Environment(loader=FileSystemLoader(template_dir))
    env.filters["display_stars"] = display_stars
    return env


@pytest.fixture()
def stars_description() -> dict[int, dict[str, str]]:
    return STARS_DESCRIPTION
