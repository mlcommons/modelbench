import pathlib

import yaml

SIMPLE_BBQ_DATA = pathlib.Path(__file__).parent / "data/full_runs/simple_bbq"
SIMPLE_TOXICITY_DATA = pathlib.Path(__file__).parent / "data/full_runs/toxicity"
from unittest.mock import Mock

import pytest

from coffee.helm_runner import (
    HelmSut,
    BbqHelmTest,
    HelmResult,
    CliHelmRunner,
    RealToxicityPromptsHelmTest,
)


def test_cli_helm_runner_command(cwd_tmpdir):
    runner = CliHelmRunner()
    runner._execute = Mock()
    runner.run([BbqHelmTest()], [HelmSut.GPT2])
    shell_arguments = runner._execute.call_args.args[0]
    runspecs = shell_arguments[shell_arguments.index("-r") + 1 :]
    assert runspecs[0] == "bbq:subject=Age,model=openai/gpt2"
    assert len(runspecs) == len(BbqHelmTest.CATEGORIES)


def test_runspec_without_params():
    r = CliHelmRunner()
    rs = r._build_runspecs([HelmSut.GPT2], [BbqHelmTest()])
    assert rs[0] == "bbq:subject=Age,model=openai/gpt2"


def test_runspec_with_params():
    r = CliHelmRunner()
    rs = r._build_runspecs([HelmSut.GPT2], [RealToxicityPromptsHelmTest()])
    assert rs[0] == "real_toxicity_prompts:model=openai/gpt2"


def test_cli_helm_runner_command_handles_huggingface_models_with_config(cwd_tmpdir):
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
    assert fb["client_spec"]["class_name"] == "helm.proxy.clients.huggingface_client.HuggingFaceClient"


@pytest.mark.datafiles(SIMPLE_BBQ_DATA)
def test_read_scores(datafiles):
    hr = HelmResult([BbqHelmTest()], [HelmSut.GPT2], datafiles, None)
    scores = hr.load_scores()
    sut_scores = scores.for_sut(HelmSut.GPT2)
    assert "BbqHelmTest" in sut_scores
    assert len(sut_scores["BbqHelmTest"]) == 2
    assert sut_scores["BbqHelmTest"]["Age"]["bbq_accuracy"] == 0.7


def test_helmsut_basics():
    assert HelmSut.GPT2.key == "openai/gpt2"
    assert hash(HelmSut.GPT2) is not None


def test_helmsut_huggingface():
    assert HelmSut.GPT2.huggingface is False
    assert HelmSut.FB_OPT_125M.huggingface is True
    assert HelmSut.PYTHIA_70M.huggingface is True
