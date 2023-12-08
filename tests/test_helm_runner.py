import pathlib

SIMPLE_BBQ_DATA = pathlib.Path(__file__).parent / "data/full_runs/simple_bbq"
from unittest.mock import Mock

import pytest

from coffee.run import (
    CliHelmRunner,
    BbqHelmTest,
    HelmSut,
    HelmResult,
    RidiculousBenchmark,
    quantize_stars,
)


def test_cli_helm_runner_command(cwd_tmpdir):
    runner = CliHelmRunner()
    runner._execute = Mock()
    runner.run([BbqHelmTest()], [HelmSut.GPT2])
    shell_arguments = runner._execute.call_args.args[0]
    assert "helm-run" == shell_arguments[0]
    runspecs = shell_arguments[shell_arguments.index("-r") + 1 :]
    assert "bbq:subject=Age,model=huggingface/gpt2" == runspecs[0]
    assert len(BbqHelmTest.CATEGORIES) == len(runspecs)


@pytest.mark.datafiles(SIMPLE_BBQ_DATA)
def test_read_scores(datafiles):
    hr = HelmResult([BbqHelmTest()], [HelmSut.GPT2], datafiles, None)
    scores = hr.load_scores()
    sut_scores = scores.for_sut(HelmSut.GPT2)
    assert "BbqHelmTest" in sut_scores
    assert 2 == len(sut_scores["BbqHelmTest"])
    assert 0.7 == sut_scores["BbqHelmTest"]["Age"]["bbq_accuracy"]


@pytest.mark.datafiles(SIMPLE_BBQ_DATA)
def test_ridiculous_benchmark(datafiles):
    hr = HelmResult([BbqHelmTest()], [HelmSut.GPT2], datafiles, None)
    scores = hr.load_scores()
    b = RidiculousBenchmark(HelmSut.GPT2, scores.for_sut(HelmSut.GPT2))
    assert 2.25 == pytest.approx(b.overall_score())


def test_quantize_stars():
    assert 0 == quantize_stars(0)
    assert 5 == quantize_stars(5)

    assert 4.5 == quantize_stars(4.5)

    assert 4.5 == quantize_stars(4.26)
    assert 4.5 == quantize_stars(4.3)
    assert 4.5 == quantize_stars(4.4)
    assert 4.5 == quantize_stars(4.6)
    assert 4.5 == quantize_stars(4.7)
    assert 4.5 == quantize_stars(4.74)
