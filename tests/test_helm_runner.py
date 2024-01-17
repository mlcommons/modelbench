import pathlib

SIMPLE_BBQ_DATA = pathlib.Path(__file__).parent / "data/full_runs/simple_bbq"
from unittest.mock import Mock

import pytest

from coffee.run import quantize_stars
from coffee.helm_runner import (
    HelmSut,
    BbqHelmTest,
    HelmResult,
    CliHelmRunner,
    InProcessHelmRunner,
)
from coffee.benchmark import RidiculousBenchmark


def test_cli_helm_runner_command(cwd_tmpdir):
    runner = CliHelmRunner()
    runner._execute = Mock()
    runner.run([BbqHelmTest()], [HelmSut.GPT2])
    shell_arguments = runner._execute.call_args.args[0]
    assert "helm-run" == shell_arguments[0]
    runspecs = shell_arguments[shell_arguments.index("-r") + 1 :]
    assert "bbq:subject=Age,model=openai/gpt2" == runspecs[0]
    assert len(BbqHelmTest.CATEGORIES) == len(runspecs)


def test_cli_helm_runner_command_handles_huggingface_models(cwd_tmpdir):
    runner = CliHelmRunner()
    runner._execute = Mock()
    # try one normal model, one magic huggingface model
    runner.run([BbqHelmTest()], [HelmSut.GPT2, HelmSut.FB_OPT_125M, HelmSut.PYTHIA_70M])
    shell_arguments = runner._execute.call_args.args[0]
    enables = [
        i for (i, s) in enumerate(shell_arguments) if s == "--enable-huggingface-models"
    ]
    assert len(enables) == 1
    assert shell_arguments[enables[0] + 1] == HelmSut.FB_OPT_125M.key
    assert shell_arguments[enables[0] + 2] == HelmSut.PYTHIA_70M.key


def test_inprocess_helm_runner(cwd_tmpdir):
    pass


@pytest.mark.datafiles(SIMPLE_BBQ_DATA)
def test_read_scores(datafiles):
    hr = HelmResult([BbqHelmTest()], [HelmSut.GPT2], datafiles)
    scores = hr.load_scores()
    sut_scores = scores.for_sut(HelmSut.GPT2)
    assert "BbqHelmTest" in sut_scores
    assert 2 == len(sut_scores["BbqHelmTest"])
    assert 0.7 == sut_scores["BbqHelmTest"]["Age"]["bbq_accuracy"]


@pytest.mark.datafiles(SIMPLE_BBQ_DATA)
def test_ridiculous_benchmark(datafiles):
    hr = HelmResult([BbqHelmTest()], [HelmSut.GPT2], datafiles)
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


def test_helmsut_basics():
    assert HelmSut.GPT2.key == "openai/gpt2"
    assert hash(HelmSut.GPT2) is not None


def test_helmsut_huggingface():
    assert HelmSut.GPT2.huggingface == False
    assert HelmSut.FB_OPT_125M.huggingface == True
    assert HelmSut.PYTHIA_70M.huggingface == True
