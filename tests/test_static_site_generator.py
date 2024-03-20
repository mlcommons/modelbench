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
    ValueEstimate,
)
from coffee.static_site_generator import StaticSiteGenerator, display_stars


@pytest.fixture()
def benchmark_score(start_time, end_time, fake_secrets):
    bd = GeneralChatBotBenchmarkDefinition(secrets=fake_secrets)
    bs = BenchmarkScore(
        bd,
        NewhelmSut.GPT2,
        [
            HazardScore(
                hazard_definition=BiasHazardDefinition(secrets=fake_secrets),
                score=ValueEstimate.make([0.5]),
                test_scores={},
            ),
            HazardScore(
                hazard_definition=ToxicityHazardDefinition(secrets=fake_secrets),
                score=ValueEstimate.make([0.8]),
                test_scores={},
            ),
        ],
        start_time=start_time,
        end_time=end_time,
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
    assert static_site_generator.benchmarks_path(page_type="benchmarks") == "benchmarks.html"
    assert static_site_generator_view_embed.benchmarks_path(page_type="benchmarks") == "../benchmarks"
    assert static_site_generator_view_embed.benchmarks_path(page_type="benchmark") == "../../benchmarks"


def test_benchmark_path(static_site_generator, static_site_generator_view_embed):
    assert (
        static_site_generator.benchmark_path("general_chat_bot_benchmark", page_type="benchmarks")
        == "general_chat_bot_benchmark.html"
    )
    assert (
        static_site_generator_view_embed.benchmark_path("general_chat_bot_benchmark", page_type="test_report")
        == "../general_chat_bot_benchmark"
    )
    assert (
        static_site_generator_view_embed.benchmark_path("general_chat_bot_benchmark", page_type="benchmarks")
        == "general_chat_bot_benchmark"
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
    def benchmark_score(self, fake_secrets):
        bd = GeneralChatBotBenchmarkDefinition(secrets=fake_secrets)
        bh = BiasHazardDefinition(secrets=fake_secrets)
        bs = BenchmarkScore(
            bd,
            NewhelmSut.GPT2,
            [
                HazardScore(hazard_definition=bh, score=ValueEstimate.make([bh.three_star_standard()]), test_scores={}),
            ],
            None,
            None,
        )
        return bs

    @pytest.fixture
    def mock_content(self, benchmark_score):
        """
        Mock the content method of StaticSiteGenerator and render each template with enough arguments to ensure
        all the partials, iterators, etc. are being exercised. Return the mocked content object so we can introspect its
        call args.
        """
        with patch("coffee.static_site_generator.StaticSiteGenerator.content") as mock_content:
            _ssg = StaticSiteGenerator()

            # We need to mock the undefined method of Environment because templates can ask for arbitrary things
            # and by returning a MagicMock, we can both satisfy any requirements on those objects as well as
            # giving us a return value we can introspect if we need to later on.
            def undefined(obj=None, name=None):
                return MagicMock(name=name)

            _ssg.env.undefined = undefined

            for template in (pathlib.Path(__file__).parent.parent / "src" / "coffee" / "templates").glob("*.html"):
                _ssg._render_template(
                    template.name,
                    benchmark_score=benchmark_score,
                    benchmark_definition=benchmark_score.benchmark_definition,
                    grouped_benchmark_scores=_ssg._grouped_benchmark_scores([benchmark_score]),
                )
            return mock_content

    @pytest.fixture(autouse=True)
    def required_template_content_keys(self, mock_content):
        """
        Return a dict with all keys requested from the calls to the mocked content object passed in. These keys are
        either a string ("general") or the name of a base class for a Sut, Benchmark, Harm, etc. These can be used
        in assertions in tests.
        """
        required_keys = dict()
        for call in mock_content.call_args_list:
            if isinstance(call.args[0], str):
                key = call.args[0]
            else:
                key = call.args[0].__class__.__base__.__name__
            if key not in required_keys:
                required_keys[key] = set()
            required_keys[key].add(call.args[1])

        return required_keys

    @pytest.mark.parametrize(
        "benchmark",
        [subclass for subclass in BenchmarkDefinition.__subclasses__()],
    )
    def test_benchmark_definitions(self, ssg, benchmark, required_template_content_keys, fake_secrets):
        for key in required_template_content_keys["BenchmarkDefinition"]:
            assert ssg.content(benchmark(secrets=fake_secrets), key)

    @pytest.mark.parametrize(
        "sut_key",
        NewhelmSut,
    )
    def test_sut_definitions(self, ssg, sut_key, required_template_content_keys):
        for key in required_template_content_keys["SutDescription"]:
            assert ssg.content(sut_key, key)

    @pytest.mark.parametrize(
        "hazard",
        [subclass for subclass in HazardDefinition.__subclasses__()],
    )
    def test_hazard_definitions(self, ssg, hazard, required_template_content_keys, fake_secrets):
        for key in required_template_content_keys["HazardDefinition"]:
            assert ssg.content(hazard(secrets=fake_secrets), key)

    def test_tests(self, ssg):
        # todo: This is the most naive version of this test, but we'll want a way to check all of the tests Coffee cares about at some point

        from newhelm.tests.bbq import BBQ

        test = BBQ()
        assert ssg.content(test, "display_name")
