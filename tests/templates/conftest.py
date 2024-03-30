import pathlib

import pytest
from jinja2 import Environment, FileSystemLoader

from coffee.benchmarks import (
    GeneralPurposeAiChatBenchmark,
    BenchmarkScore,
)
from coffee.scoring import ValueEstimate
from coffee.hazards import HazardScore, SafeCaeHazard, SafeCbrHazard
from coffee.newhelm_runner import NewhelmSut
from coffee.static_site_generator import (
    StaticSiteGenerator,
    min_bar_size,
)
from coffee.utilities import group_by_key


def _benchmark_score(end_time) -> BenchmarkScore:
    bd = GeneralPurposeAiChatBenchmark()
    cae_hazard = SafeCaeHazard()
    cbr_hazard = SafeCbrHazard()
    cae_score = HazardScore(hazard_definition=cae_hazard, score=ValueEstimate.make(0.5, 10), test_scores={})
    cbr_score = HazardScore(hazard_definition=cbr_hazard, score=ValueEstimate.make(0.5, 10), test_scores={})
    bs = BenchmarkScore(
        bd,
        NewhelmSut.GPT2,
        [cae_score, cbr_score],
        end_time,
    )
    return bs


@pytest.fixture()
def benchmark_score(end_time) -> BenchmarkScore:
    return _benchmark_score(end_time)


@pytest.fixture()
def grouped_benchmark_scores(end_time) -> dict[str, list[BenchmarkScore]]:
    scores = [_benchmark_score(end_time)]
    return group_by_key(scores, key=lambda x: x.benchmark_definition)


def _template_env(view_embed: bool = False) -> Environment:
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
    ssg = StaticSiteGenerator(view_embed=view_embed)
    env.filters["min_bar_size"] = min_bar_size
    env.globals["root_path"] = ssg.root_path
    env.globals["benchmarks_path"] = ssg.benchmarks_path
    env.globals["benchmark_path"] = ssg.benchmark_path
    env.globals["test_report_path"] = ssg.test_report_path
    ssg._content = update_dict_values(ssg._content)
    env.globals["content"] = ssg.content
    return env


@pytest.fixture()
def template_env() -> Environment:
    return _template_env()


@pytest.fixture()
def template_env_view_embed() -> Environment:
    return _template_env(view_embed=True)
