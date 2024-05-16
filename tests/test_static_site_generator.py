import abc
import datetime
import pathlib
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from modelbench.benchmarks import (
    BenchmarkDefinition,
    GeneralPurposeAiChatBenchmark,
    BenchmarkScore,
)
from modelbench.hazards import HazardScore, SafeCaeHazard, SafeCbrHazard, SafeHazard
from modelbench.modelgauge_runner import ModelGaugeSut, SutDescription
from modelbench.safe_test import SafeTest, PersonaType
from modelbench.scoring import ValueEstimate
from modelbench.static_site_generator import HazardScorePositions, StaticSiteGenerator


@pytest.fixture()
def benchmark_score(end_time):
    bd = GeneralPurposeAiChatBenchmark()
    bs = BenchmarkScore(
        bd,
        ModelGaugeSut.ALPACA_7B,
        [
            HazardScore(
                hazard_definition=SafeCaeHazard(),
                score=ValueEstimate.make(0.5, 10),
                test_scores={},
            ),
            HazardScore(
                hazard_definition=SafeCbrHazard(),
                score=ValueEstimate.make(0.8, 20),
                test_scores={},
            ),
        ],
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
        "general_purpose_ai_chat_benchmark.html",
        "static/images/ml_commons_logo.png",
        "static/style.css",
        "benchmarks.html",
        "general_purpose_ai_chat_benchmark.html",
        "alpaca-7b_general_purpose_ai_chat_benchmark_report.html",
        "index.html",
    ],
)
def test_creates_files(web_dir, path):
    assert (web_dir / path).exists()


def test_root_path(static_site_generator, static_site_generator_view_embed):
    assert static_site_generator.root_path() == "index.html"
    assert static_site_generator_view_embed.root_path() == "#"


def test_benchmarks_path(static_site_generator, static_site_generator_view_embed):
    assert static_site_generator.benchmarks_path(page_type="benchmarks") == "benchmarks.html"
    assert static_site_generator_view_embed.benchmarks_path(page_type="benchmarks") == "../ai-safety"
    assert static_site_generator_view_embed.benchmarks_path(page_type="benchmark") == "../../ai-safety"


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
    Tests to ensure that appropriate presentation-layer content exists for objects that are added to modelbench.
    """

    @pytest.fixture
    def ssg(self):
        _ssg = StaticSiteGenerator()
        return _ssg

    @pytest.fixture
    def benchmark_score(self):
        bd = GeneralPurposeAiChatBenchmark()
        bh = SafeCaeHazard()
        bs = BenchmarkScore(
            bd,
            ModelGaugeSut.ALPACA_7B,
            [
                HazardScore(
                    hazard_definition=bh, score=ValueEstimate.make(bh.reference_standard(), 50), test_scores={}
                ),
            ],
            datetime.datetime.fromtimestamp(170000000),
        )
        return bs

    @pytest.fixture
    def mock_content(self, benchmark_score):
        """
        Mock the content method of StaticSiteGenerator and render each template with enough arguments to ensure
        all the partials, iterators, etc. are being exercised. Return the mocked content object so we can introspect its
        call args.
        """
        with patch("modelbench.static_site_generator.StaticSiteGenerator.content") as mock_content:
            _ssg = StaticSiteGenerator()

            # We need to mock the undefined method of Environment because templates can ask for arbitrary things
            # and by returning a MagicMock, we can both satisfy any requirements on those objects as well as
            # giving us a return value we can introspect if we need to later on.
            def undefined(obj=None, name=None):
                return MagicMock(name=name)

            _ssg.env.undefined = undefined

            for template in (pathlib.Path(__file__).parent.parent / "src" / "modelbench" / "templates").glob("*.html"):
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
    def test_benchmark_definitions(self, ssg, benchmark, required_template_content_keys):
        for key in required_template_content_keys["BenchmarkDefinition"]:
            assert ssg.content(benchmark(), key)

    @pytest.mark.parametrize(
        "sut_key",
        [sut.key for sut in ModelGaugeSut],
    )
    def test_sut_definitions(self, ssg, sut_key, required_template_content_keys):
        for key in required_template_content_keys["SutDescription"]:
            assert ssg.content(sut_key, key)

    @pytest.mark.parametrize(
        "hazard",
        [subclass for subclass in SafeHazard.__subclasses__() if not abc.ABC in subclass.__bases__],
    )
    def test_safe_hazard_definitions(self, ssg, hazard, required_template_content_keys):
        for key in required_template_content_keys["SafeHazard"]:
            assert ssg.content(hazard(), key)

    def test_tests(self, ssg):
        # todo: This is the most naive version of this test, but we'll want a way to check all of the tests modelbench cares about at some point

        test = SafeTest(uid="mls_ais_wg_safe_ssh_hazardous", hazard="nvc", folder_name="foo", together_api_key=None,
                        persona_types=[PersonaType.TYPICAL])
        assert ssg.content(test, "display_name") == "SafeTest: Suicide, self-harm and eating disorders (Hazardous)"

    def test_test_defaults(self, ssg):
        test = SafeTest(uid="not_a_real_uid", hazard="nvc", folder_name="foo", together_api_key=None,
                        persona_types=[PersonaType.TYPICAL])
        assert ssg.content(test, "display_name") == "not_a_real_uid"
        assert ssg.content(test, "not_a_real_key") == ""


def test_sut_content_defaults():
    ssg = StaticSiteGenerator()
    a_dynamic_sut = SutDescription("fake", "Fake SUT")
    assert ssg.content(a_dynamic_sut, "name") == "Fake SUT"


class TestHazardScorePositions:
    @pytest.fixture
    def hazard_score(self):
        def func(probability):
            cae_hazard = SafeCaeHazard()
            hs = HazardScore(hazard_definition=cae_hazard, score=ValueEstimate.make(probability, 5000), test_scores={})
            return hs

        return func

    def test_grade_bands(self, hazard_score):
        hs = hazard_score(0.5)
        hsp = HazardScorePositions(lowest_bar_percent=1.0, min_bar_width=0.0)
        assert [x[1] for x in hsp(hs)["grade_bands"]] == [x * 100 for x in hs.grade_points()[1:]]
        assert sum([x[1] - x[0] for x in hsp(hs)["grade_bands"]]) == 100

    def test_grade_bands_min_bar_width(self, hazard_score):
        hs = hazard_score(0.5)
        hsp = HazardScorePositions(lowest_bar_percent=0.8, min_bar_width=0.02)
        assert hsp(hs)["grade_bands"] == [
            (0.0, pytest.approx(77.0341)),
            (pytest.approx(77.0341), 94.0),
            (94.0, 96.0),
            (96.0, 98.0),
            (98.0, 100.0),
        ]
        assert sum([x[1] - x[0] for x in hsp(hs)["grade_bands"]]) == 100

    def test_grade_bands_lowest_bar_percent(self, hazard_score):
        hs = hazard_score(0.5)
        hsp = HazardScorePositions(lowest_bar_percent=0.5, min_bar_width=0.02)
        assert hsp(hs)["grade_bands"] == [
            (0.0, pytest.approx(48.14634)),
            (pytest.approx(48.14634), 94.0),
            (94.0, 96.0),
            (96.0, 98.0),
            (98.0, 100.0),
        ]
        assert sum([x[1] - x[0] for x in hsp(hs)["grade_bands"]]) == 100

    def test_point_position(self, hazard_score):
        hs = hazard_score(0.5)
        hsp = HazardScorePositions(lowest_bar_percent=1, min_bar_width=0)
        assert hsp(hs)["point_position"] == pytest.approx(50)

    def test_point_position_lowest_bar_percent(self, hazard_score):
        hs = hazard_score(0.5)
        hsp = HazardScorePositions(lowest_bar_percent=0.5, min_bar_width=0)
        assert hsp(hs)["point_position"] == pytest.approx(25)

    @pytest.mark.parametrize("lowest_bar_percent", [1.0, 0.5])
    @pytest.mark.parametrize("min_bar_width", [0.02, 0.02, 0.04])
    @pytest.mark.parametrize("probability", [0.5, 0.001, 0.999])
    def test_point_does_not_exceed_bounds(self, lowest_bar_percent, min_bar_width, probability, hazard_score):
        hs = hazard_score(probability)
        hsp = HazardScorePositions(lowest_bar_percent=lowest_bar_percent, min_bar_width=min_bar_width)
        bounds = hsp(hs)["grade_bands"][hs.numeric_grade() - 1]
        assert bounds[0] <= hsp(hs)["point_position"] <= bounds[1]

    def test_error_bar(self, hazard_score):
        hs = hazard_score(0.5)
        hsp = HazardScorePositions(min_bar_width=0.04, lowest_bar_percent=0.5)
        assert hsp(hs)["error_bar"]["start"] == pytest.approx(24.30221)
        assert hsp(hs)["error_bar"]["width"] == pytest.approx(1.395562)
