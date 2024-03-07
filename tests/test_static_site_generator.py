import pathlib
from unittest.mock import MagicMock
from unittest.mock import patch

SIMPLE_BBQ_DATA = pathlib.Path(__file__).parent / "data/full_runs/simple_bbq"

import pytest

from coffee.newhelm_runner import NewhelmSut
from coffee.benchmark import (
    BenchmarkDefinition,
    GeneralChatBotBenchmarkDefinition,
    BiasHazardDefinition,
    HazardDefinition,
    HazardScore,
    BenchmarkScore,
    ToxicityHazardDefinition,
)
from coffee.static_site_generator import StaticSiteGenerator, display_stars


@pytest.fixture()
def benchmark_score(start_time, end_time):
    bd = GeneralChatBotBenchmarkDefinition()
    bs = BenchmarkScore(
        bd,
        NewhelmSut.GPT2,
        [
            HazardScore(BiasHazardDefinition(), 0.5, start_time, end_time),
            HazardScore(ToxicityHazardDefinition(), 0.8, start_time, end_time),
        ],
        start_time,
        end_time,
    )
    return bs


@pytest.fixture()
def web_dir(tmp_path, benchmark_score):
    generator = StaticSiteGenerator()
    generator.generate([benchmark_score], tmp_path)
    return tmp_path


@pytest.fixture()
def static_site_generator():
    return StaticSiteGenerator()


@pytest.fixture()
def static_site_generator_view_embed():
    return StaticSiteGenerator(view_embed=True)


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
def test_creates_files(web_dir, path):
    assert (web_dir / path).exists()


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


def test_root_path(static_site_generator, static_site_generator_view_embed):
    assert static_site_generator.root_path() == "index.html"
    assert static_site_generator_view_embed.root_path() == "#"


def test_benchmarks_path(static_site_generator, static_site_generator_view_embed):
    assert static_site_generator.benchmarks_path() == "benchmarks.html"
    assert static_site_generator_view_embed.benchmarks_path() == "../benchmarks"


def test_benchmark_path(static_site_generator, static_site_generator_view_embed):
    assert static_site_generator.benchmark_path("general_chat_bot_benchmark") == "general_chat_bot_benchmark.html"
    assert (
        static_site_generator_view_embed.benchmark_path("general_chat_bot_benchmark") == "../general_chat_bot_benchmark"
    )


class TestObjectContentKeysExist:
    """
    Tests to ensure that appropriate presentation-layer content exists for objects that are added to coffee.
    """

    @pytest.fixture
    def ssg(self):
        _ssg = StaticSiteGenerator()
        return _ssg

    @pytest.fixture
    def benchmark_score(self):
        bd = GeneralChatBotBenchmarkDefinition()
        bh = BiasHazardDefinition()
        bs = BenchmarkScore(
            bd,
            NewhelmSut.GPT2,
            [
                HazardScore(bh, bh.three_star_standard(), None, None),
            ],
            None,
            None,
        )
        return bs

    @pytest.fixture(autouse=True)
    def mock_content(self, benchmark_score):
        with patch("coffee.static_site_generator.StaticSiteGenerator.content") as mock_content:
            _ssg = StaticSiteGenerator()

            def undefined(obj=None, name=None):
                return MagicMock(name=name)

            _ssg.env.undefined = undefined

            for template in ["benchmarks.html", "benchmark.html", "test_report.html"]:
                _ssg._render_template(
                    template,
                    benchmark_score=benchmark_score,
                    benchmark_definition=benchmark_score.benchmark_definition,
                    grouped_benchmark_scores=_ssg._grouped_benchmark_scores([benchmark_score]),
                )

            all_calls = dict()
            for call in mock_content.call_args_list:
                if isinstance(call.args[0], str):
                    key = call.args[0]
                else:
                    key = call.args[0].__class__.__base__.__name__
                if key not in all_calls:
                    all_calls[key] = set()
                all_calls[key].add(call.args[1])

            return all_calls

    @pytest.mark.parametrize(
        "sc",
        BenchmarkDefinition.__subclasses__(),
    )
    def test_benchmark_definitions(self, ssg, sc, mock_content):
        for key in mock_content["BenchmarkDefinition"]:
            assert ssg.content(sc(), key)

    @pytest.mark.parametrize(
        "item",
        NewhelmSut,
    )
    def test_sut_definitions(self, ssg, item, mock_content):
        for key in mock_content["SutDescription"]:
            assert ssg.content(item, key)

    @pytest.mark.parametrize(
        "sc",
        HazardDefinition.__subclasses__(),
    )
    def test_hazard_definitions(self, ssg, sc, mock_content):
        for key in mock_content["HazardDefinition"]:
            assert ssg.content(sc(), key)
