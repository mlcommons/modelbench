import json
import pathlib
import unittest.mock
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from modelbench.benchmarks import BenchmarkDefinition
from modelbench.hazards import HazardScore, SafeCbrHazard
from modelbench.hazards import SafeHazard
from modelbench.modelgauge_runner import ModelGaugeSut
from modelbench.run import benchmark, cli, find_suts_for_sut_argument, update_standards_to
from modelbench.scoring import ValueEstimate


@patch("modelbench.run.run_tests")
def test_update_standards(fake_run, tmp_path, fake_secrets):
    with unittest.mock.patch("modelbench.run.load_secrets_from_config", return_value=fake_secrets):
        bias_hazard = SafeCbrHazard()
        fake_run.return_value = {
            bias_hazard: HazardScore(
                hazard_definition=bias_hazard, score=ValueEstimate.make(0.123456, 100), test_scores={}
            )
        }
        new_path = pathlib.Path(tmp_path) / "standards.json"
        update_standards_to(new_path)
        assert new_path.exists()
        with open(new_path) as f:
            j = json.load(f)
            assert j["standards"]["reference_standards"][bias_hazard.key()] == 0.123456
            assert j["standards"]["reference_suts"][0]["id"] == "vicuna-13b"


def test_find_suts():
    # nothing gets everything
    assert find_suts_for_sut_argument([]) == ModelGaugeSut

    # key from modelbench gets a known SUT
    assert find_suts_for_sut_argument(["alpaca-7b"]) == [ModelGaugeSut.ALPACA_7B]

    # key from modelgauge gets a dynamic one
    dynamic_qwen = find_suts_for_sut_argument(["Qwen1.5-72B-Chat"])[0]
    assert dynamic_qwen.key == "Qwen1.5-72B-Chat"
    assert dynamic_qwen.display_name == "Qwen1.5 72B Chat"

    with pytest.raises(click.BadParameter):
        find_suts_for_sut_argument(["something nonexistent"])


class TestCli:

    @pytest.fixture(autouse=True)
    def mock_score_benchmarks(self, monkeypatch):
        import modelbench

        mock_obj = MagicMock()

        monkeypatch.setattr(modelbench.run, "score_benchmark", mock_obj)
        return mock_obj

    @pytest.fixture(autouse=True)
    def do_not_make_static_site(self, monkeypatch):
        import modelbench

        monkeypatch.setattr(modelbench.run, "generate_content", MagicMock())

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_nonexistent_benchmarks_can_not_be_called(self, runner):
        result = runner.invoke(cli, ["benchmark", "--benchmark", "NotARealBenchmark"])
        assert result.exit_code == 2
        assert "Invalid value for '--benchmark'" in result.output

    def test_calls_score_benchmark_with_correct_benchmark(self, runner, mock_score_benchmarks):
        class MyBenchmark(BenchmarkDefinition):
            def __init__(self):
                super().__init__([c() for c in SafeHazard.__subclasses__()])

        cli.commands["benchmark"].params[-2].type.choices += ["MyBenchmark"]
        result = runner.invoke(cli, ["benchmark", "--benchmark", "MyBenchmark"])
        assert isinstance(mock_score_benchmarks.call_args.args[0], MyBenchmark)
