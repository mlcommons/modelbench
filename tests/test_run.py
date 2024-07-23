import json
import pathlib
import unittest.mock
from datetime import datetime
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from modelbench.benchmarks import BenchmarkDefinition, BenchmarkScore, GeneralPurposeAiChatBenchmark
from modelbench.hazards import HazardScore, SafeCbrHazard
from modelbench.hazards import SafeHazard
from modelbench.modelgauge_runner import ModelGaugeSut, SutDescription
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

    def mock_score(self):
        benchmark = GeneralPurposeAiChatBenchmark()
        return BenchmarkScore(
            benchmark,
            ModelGaugeSut.ALPACA_7B,
            [
                HazardScore(
                    hazard_definition=benchmark.hazards()[0], score=ValueEstimate.make(0.123456, 100), test_scores={}
                ),
            ],
            datetime.now(),
        )

    @pytest.fixture(autouse=True)
    def mock_score_benchmarks(self, monkeypatch):
        import modelbench

        mock = MagicMock(return_value=[self.mock_score()])
        monkeypatch.setattr(modelbench.run, "score_benchmarks", mock)
        return mock

    @pytest.fixture(autouse=True)
    def do_not_make_static_site(self, monkeypatch):
        import modelbench

        monkeypatch.setattr(modelbench.run, "generate_content", MagicMock())

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_benchmark_basic_run_produces_json(self, runner, tmp_path):
        with unittest.mock.patch("modelbench.run.find_suts_for_sut_argument") as mock_find_suts:
            mock_find_suts.return_value = [SutDescription("fake", "Fake Sut")]
            result = runner.invoke(
                cli,
                ["benchmark", "-m", "1", "--sut", "fake", "--output-dir", str(tmp_path.absolute())],
                catch_exceptions=False,
            )
            assert result.exit_code == 0
            assert (tmp_path / f"benchmark_record-{GeneralPurposeAiChatBenchmark().uid}.json").exists

    def test_benchmark_anonymous_run_produces_json(self, runner, tmp_path):
        with unittest.mock.patch("modelbench.run.find_suts_for_sut_argument") as mock_find_suts:
            mock_find_suts.return_value = [SutDescription("fake", "Fake Sut")]
            result = runner.invoke(
                cli,
                [
                    "benchmark",
                    "--anonymize",
                    "42",
                    "-m",
                    "1",
                    "--sut",
                    "fake",
                    "--output-dir",
                    str(tmp_path.absolute()),
                ],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.stdout
            assert (tmp_path / f"benchmark_record-{GeneralPurposeAiChatBenchmark().uid}.json").exists

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
        assert isinstance(mock_score_benchmarks.call_args.args[0][0], MyBenchmark)
