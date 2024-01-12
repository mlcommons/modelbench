import pathlib

SIMPLE_BBQ_DATA = pathlib.Path(__file__).parent / "data/full_runs/simple_bbq"

import pytest

from coffee.helm import HelmResult, BbqHelmTest, HelmSut
from coffee.benchmark import RidiculousBenchmark
from coffee.static_site_generator import StaticSiteGenerator


@pytest.fixture()
def benchmark(datafiles):
    hr = HelmResult([BbqHelmTest()], [HelmSut.GPT2], datafiles, None)
    scores = hr.load_scores()
    b = RidiculousBenchmark(HelmSut.GPT2, scores.for_sut(HelmSut.GPT2))
    return b


@pytest.mark.parametrize(
    "path",
    [
        "ridiculous_benchmark.html",
        "static/images/ml_commons_logo.png",
        "benchmarks.html",
    ],
)
@pytest.mark.datafiles(SIMPLE_BBQ_DATA)
def test_creates_files(benchmark, tmp_path, path):
    generator = StaticSiteGenerator()
    generator.generate([benchmark], tmp_path)
    assert (tmp_path / path).exists()


@pytest.mark.datafiles(SIMPLE_BBQ_DATA)
@pytest.mark.parametrize(
    "score,expected",
    [
        (2.0, (2, False, 3)),
        (2.49, (2, False, 3)),
        (2.50, (2, True, 2)),
        (2.51, (2, True, 2)),
        (4.0, (4, False, 1)),
    ],
)
def test_displays_correct_stars(benchmark, cwd_tmpdir, monkeypatch, score, expected):
    monkeypatch.setattr(benchmark, "overall_score", lambda: score)
    generator = StaticSiteGenerator()
    stars = generator.calculate_stars(benchmark)
    assert stars == expected
