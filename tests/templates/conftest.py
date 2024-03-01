import pathlib
from itertools import groupby

import pytest
from jinja2 import Environment, FileSystemLoader

from coffee.benchmark import (
    GeneralChatBotBenchmarkDefinition,
    BiasHazardDefinition,
    HazardScore,
    BenchmarkScore,
    ToxicityHazardDefinition,
)
from coffee.newhelm_runner import NewhelmSut
from coffee.static_site_generator import (
    StaticSiteGenerator,
    STARS_DESCRIPTION,
    display_stars,
)


def _benchmark_score() -> BenchmarkScore:
    bd = GeneralChatBotBenchmarkDefinition()
    bh = BiasHazardDefinition()
    th = ToxicityHazardDefinition()
    bs = BenchmarkScore(
        bd,
        NewhelmSut.GPT2,
        [
            HazardScore(bh, bh.three_star_standard()),
            HazardScore(th, th.three_star_standard()),
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
    ssg = StaticSiteGenerator()
    env.filters["display_stars"] = display_stars
    env.globals["root_path"] = ssg.root_path
    env.globals["benchmarks_path"] = ssg.benchmarks_path
    env.globals["benchmark_path"] = ssg.benchmark_path
    env.globals["test_report_path"] = ssg.test_report_path
    return env


@pytest.fixture()
def stars_description() -> dict[int, dict[str, str]]:
    return STARS_DESCRIPTION
