import pathlib

import pytest
from jinja2 import Environment, FileSystemLoader

from modelbench.benchmarks import (
    GeneralPurposeAiChatBenchmark,
    BenchmarkScore,
)
from modelbench.hazards import HazardScore, SafeCaeHazard, SafeCbrHazard
from modelbench.scoring import ValueEstimate
from modelbench.static_site_generator import (
    HazardScorePositions,
    StaticSiteGenerator,
)
from modelbench.suts import ModelGaugeSut
from modelbench.utilities import group_by_key


def _benchmark_score(end_time) -> BenchmarkScore:
    bd = GeneralPurposeAiChatBenchmark()
    cae_hazard = SafeCaeHazard()
    cbr_hazard = SafeCbrHazard()
    cae_score = HazardScore(hazard_definition=cae_hazard, score=ValueEstimate.make(0.5, 10), test_scores={})
    cbr_score = HazardScore(hazard_definition=cbr_hazard, score=ValueEstimate.make(0.5, 10), test_scores={})
    bs = BenchmarkScore(
        bd,
        ModelGaugeSut.for_key("alpaca-7b"),
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


def _template_env(view_embed: bool = False, custom_branding: pathlib.Path = None) -> Environment:
    def update_dict_values(d: dict, parent_keys=[]) -> dict:
        for k, v in d.items():
            new_keys = parent_keys + [k]
            if isinstance(v, dict):
                update_dict_values(v, new_keys)
            else:
                d[k] = "__test__." + ".".join(new_keys)
        return d

    template_dir = pathlib.Path(__file__).parent.parent.parent / "src" / "modelbench" / "templates"
    env = Environment(loader=FileSystemLoader(template_dir))
    ssg = StaticSiteGenerator(view_embed=view_embed, custom_branding=custom_branding)
    env.globals["hsp"] = HazardScorePositions(min_bar_width=0.04, lowest_bar_percent=0.5)
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


@pytest.fixture()
def template_env_mlc() -> Environment:
    return _template_env(
        custom_branding=pathlib.Path(__file__).parent.parent.parent / "src" / "modelbench" / "templates" / "content_mlc"
    )
