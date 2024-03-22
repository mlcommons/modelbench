import pathlib

import pytest
from jinja2 import Environment, FileSystemLoader

from coffee.benchmark import (
    GeneralChatBotBenchmarkDefinition,
    BiasHazardDefinition,
    HazardScore,
    BenchmarkScore,
    ToxicityHazardDefinition,
    ValueEstimate,
)
from coffee.newhelm_runner import NewhelmSut
from coffee.static_site_generator import (
    StaticSiteGenerator,
    display_stars,
)
from coffee.utilities import group_by_key


def _benchmark_score(start_time, end_time) -> BenchmarkScore:
    bd = GeneralChatBotBenchmarkDefinition()
    bh = BiasHazardDefinition()
    th = ToxicityHazardDefinition()
    bias_score = HazardScore(hazard=bh, score=ValueEstimate.make(0.5, 10), test_scores={})
    toxicity_score = HazardScore(hazard=th, score=ValueEstimate.make(0.5, 10), test_scores={})
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
    scores = [_benchmark_score(start_time, end_time)]
    return group_by_key(scores, key=lambda x: x.benchmark_definition)


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
