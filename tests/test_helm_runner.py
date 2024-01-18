import pathlib

import yaml

SIMPLE_BBQ_DATA = pathlib.Path(__file__).parent / "data/full_runs/simple_bbq"
SIMPLE_TOXICITY_DATA = pathlib.Path(__file__).parent / "data/full_runs/toxicity"
from unittest.mock import Mock

import pytest

from coffee.run import quantize_stars
from coffee.helm import (
    HelmSut,
    BbqHelmTest,
    HelmResult,
    CliHelmRunner,
    RealToxicityPromptsHelmTest,
)
from coffee.benchmark import MakeshiftBiasBenchmark, MakeshiftToxicityBenchmark


def test_cli_helm_runner_command(cwd_tmpdir):
    runner = CliHelmRunner()
    runner._execute = Mock()
    runner.run([BbqHelmTest()], [HelmSut.GPT2])
    shell_arguments = runner._execute.call_args.args[0]
    runspecs = shell_arguments[shell_arguments.index("-r") + 1 :]
    assert "bbq:subject=Age,model=openai/gpt2" == runspecs[0]
    assert len(BbqHelmTest.CATEGORIES) == len(runspecs)


def test_runspec_without_params():
    r = CliHelmRunner()
    rs = r._build_runspecs([HelmSut.GPT2], [BbqHelmTest()])
    assert rs[0] == "bbq:subject=Age,model=openai/gpt2"


def test_runspec_with_params():
    r = CliHelmRunner()
    rs = r._build_runspecs([HelmSut.GPT2], [RealToxicityPromptsHelmTest()])
    assert rs[0] == "real_toxicity_prompts:model=openai/gpt2"


def test_cli_helm_runner_command_handles_huggingface_models_with_fnord(cwd_tmpdir):
    runner = CliHelmRunner()
    runner._execute = Mock()
    # try one normal model, one magic huggingface model
    runner.run([BbqHelmTest()], [HelmSut.GPT2, HelmSut.FB_OPT_125M, HelmSut.PYTHIA_70M])
    shell_arguments = runner._execute.call_args.args[0]
    assert "--enable-huggingface-models" not in shell_arguments  # this was the old way
    md_file = pathlib.Path(cwd_tmpdir) / "run" / "prod_env" / "model_deployments.yaml"
    assert md_file.exists()
    with open(md_file) as input:
        y = yaml.safe_load(input)
    root = y["model_deployments"]
    assert len(root) == 2
    fb = root[0]
    assert fb["name"] == HelmSut.FB_OPT_125M.key
    assert fb["tokenizer_name"] == HelmSut.FB_OPT_125M.tokenizer_name
    assert fb["max_sequence_length"] == HelmSut.FB_OPT_125M.tokenizer_max_length
    assert (
        fb["client_spec"]["class_name"]
        == "helm.proxy.clients.huggingface_client.HuggingFaceClient"
    )


@pytest.mark.datafiles(SIMPLE_BBQ_DATA)
def test_read_scores(datafiles):
    hr = HelmResult([BbqHelmTest()], [HelmSut.GPT2], datafiles, None)
    scores = hr.load_scores()
    sut_scores = scores.for_sut(HelmSut.GPT2)
    assert "BbqHelmTest" in sut_scores
    assert 2 == len(sut_scores["BbqHelmTest"])
    assert 0.7 == sut_scores["BbqHelmTest"]["Age"]["bbq_accuracy"]


@pytest.mark.datafiles(SIMPLE_BBQ_DATA)
def test_makeshift_bias_benchmark(datafiles):
    hr = HelmResult([BbqHelmTest()], [HelmSut.GPT2], datafiles, None)
    scores = hr.load_scores()
    b = MakeshiftBiasBenchmark(HelmSut.GPT2, scores.for_sut(HelmSut.GPT2))
    assert 2.25 == pytest.approx(b.overall_score())


@pytest.mark.datafiles(SIMPLE_TOXICITY_DATA)
def test_makeshift_toxicity_benchmark(datafiles):
    hr = HelmResult([RealToxicityPromptsHelmTest()], [HelmSut.GPT2], datafiles, None)
    scores = hr.load_scores()
    b = MakeshiftToxicityBenchmark(HelmSut.GPT2, scores.for_sut(HelmSut.GPT2))
    assert 4.3 == pytest.approx(b.overall_score())


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
