import pathlib
from itertools import groupby

import pytest
from jinja2 import Environment, FileSystemLoader

from coffee.benchmark import (
    GeneralChatBotBenchmarkDefinition,
    Hazard,
    HazardScore,
    BenchmarkScore,
    ValueEstimate,
)
from coffee.newhelm_runner import NewhelmSut
from coffee.static_site_generator import (
    StaticSiteGenerator,
    display_stars,
)


def _benchmark_score(start_time, end_time) -> BenchmarkScore:
    bd = GeneralChatBotBenchmarkDefinition()

    bias_score = HazardScore(hazard=Hazard.BIAS, score=ValueEstimate.make([0.5]), test_scores={})
    toxicity_score = HazardScore(hazard=Hazard.TOXICITY, score=ValueEstimate.make([0.5]), test_scores={})
    bs = BenchmarkScore(
        bd,
        NewhelmSut.GPT2,
        [bias_score, toxicity_score],
        start_time,
        end_time,
    )
    return bs


@pytest.fixture()
def benchmark_score(start_time, end_time) -> BenchmarkScore:
    return _benchmark_score(start_time, end_time)


@pytest.fixture()
def grouped_benchmark_scores(start_time, end_time) -> dict[str, list[BenchmarkScore]]:
    benchmark_scores_dict = {}
    for benchmark_definition, grouped_benchmark_scores in groupby(
        [_benchmark_score(start_time, end_time)], lambda x: x.benchmark_definition
    ):
        grouped_benchmark_scores_list: list = list(grouped_benchmark_scores)
        benchmark_scores_dict[benchmark_definition] = grouped_benchmark_scores_list
    return benchmark_scores_dict


@pytest.fixture()
def template_env() -> Environment:
    def update_dict_values(d: dict, parent_keys=[]) -> dict:
        for k, v in d.items():
            new_keys = parent_keys + [k]
            if isinstance(v, dict):
                update_dict_values(v, new_keys)
            else:
                d[k] = "__test__." + ".".join(new_keys)
        return d

    template_dir = pathlib.Path(__file__).parent.parent.parent / "src" / "coffee" / "templates"
    env = Environment(loader=FileSystemLoader(template_dir))
    ssg = StaticSiteGenerator()
    env.filters["display_stars"] = display_stars
    env.globals["root_path"] = ssg.root_path
    env.globals["benchmarks_path"] = ssg.benchmarks_path
    env.globals["benchmark_path"] = ssg.benchmark_path
    env.globals["test_report_path"] = ssg.test_report_path
    ssg._content = update_dict_values(ssg._content)
    env.globals["content"] = ssg.content
    return env
