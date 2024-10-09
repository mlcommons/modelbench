import json
import pathlib
import unittest.mock
from datetime import datetime
from typing import Sequence, Mapping, List
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from modelgauge.base_test import PromptResponseTest
from modelgauge.records import TestRecord
from modelgauge.secret_values import RawSecrets
from modelgauge.tests.safe_v1 import Locale

from modelbench.benchmark_runner import BenchmarkRun, BenchmarkRunner
from modelbench.benchmarks import (
    BenchmarkDefinition,
    BenchmarkScore,
    GeneralPurposeAiChatBenchmark,
    GeneralPurposeAiChatBenchmarkV1,
)
from modelbench.hazards import HazardScore, HazardDefinition
from modelbench.hazards import SafeHazard
from modelbench.run import benchmark, cli, find_suts_for_sut_argument, get_benchmark, update_standards_to
from modelbench.scoring import ValueEstimate
from modelbench.suts import SutDescription, SUTS_FOR_V_0_5, ModelGaugeSut


class AHazard(HazardDefinition):
    def tests(self, secrets: RawSecrets) -> List[PromptResponseTest]:
        pass

    def score(self, sut_scores: Mapping[str, TestRecord]) -> "HazardScore":
        return HazardScore(
            hazard_definition=self, score=ValueEstimate.make(0.123456, 100), test_scores={}, exceptions=0
        )


def fake_benchmark_run(hazard, tmp_path):
    sut = ModelGaugeSut.for_key("mistral-7b")

    class ABenchmark(BenchmarkDefinition):
        def _make_hazards(self) -> Sequence[HazardDefinition]:
            return [hazard]

    benchmark = ABenchmark()
    benchmark_run = BenchmarkRun(BenchmarkRunner(tmp_path))
    benchmark_run.benchmarks = [benchmark]
    benchmark_run.benchmark_scores[benchmark][sut] = BenchmarkScore(benchmark, sut, [hazard.score({})], None)
    return benchmark_run


@patch("modelbench.run.run_benchmarks_for_suts")
def test_update_standards(fake_runner, tmp_path, fake_secrets):
    with unittest.mock.patch("modelbench.run.load_secrets_from_config", return_value=fake_secrets):
        hazard = AHazard()
        benchmark_run = fake_benchmark_run(hazard, tmp_path)
        fake_runner.return_value = benchmark_run

        new_path = pathlib.Path(tmp_path) / "standards.json"
        update_standards_to(new_path)
        assert new_path.exists()
        with open(new_path) as f:
            j = json.load(f)
            print(j)
            assert j["standards"]["reference_standards"][hazard.uid] == 0.123456
            assert j["standards"]["reference_suts"][0] == "mistral-7b"


def test_find_suts():
    # nothing gets everything
    assert find_suts_for_sut_argument([]) == SUTS_FOR_V_0_5

    # key from modelbench gets a known SUT
    assert find_suts_for_sut_argument(["mistral-7b"]) == [ModelGaugeSut.for_key("mistral-7b")]

    # key from modelgauge gets a dynamic one
    dynamic_qwen = find_suts_for_sut_argument(["Qwen1.5-72B-Chat"])[0]
    assert dynamic_qwen.key == "Qwen1.5-72B-Chat"

    with pytest.raises(click.BadParameter):
        find_suts_for_sut_argument(["something nonexistent"])


class TestCli:

    class MyBenchmark(BenchmarkDefinition):

        def _make_hazards(self) -> Sequence[HazardDefinition]:
            return [c() for c in SafeHazard.__subclasses__()]

        @property
        def uid(self):
            return "my_benchmark"

    def mock_score(self):
        benchmark = GeneralPurposeAiChatBenchmark()
        return BenchmarkScore(
            benchmark,
            ModelGaugeSut.for_key("mistral-7b"),
            [
                HazardScore(
                    hazard_definition=benchmark.hazards()[0],
                    score=ValueEstimate.make(0.123456, 100),
                    test_scores={},
                    exceptions=0,
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

    @pytest.mark.parametrize(
        "version,locale", [("0.5", None), ("1.0", "en_US"), ("1.0", "fr_FR"), ("1.0", "hi_IN"), ("1.0", "zh_CN")]
    )
    def test_benchmark_basic_run_produces_json(self, runner, version, locale, tmp_path):
        benchmark_options = ["--version", version]
        if locale is not None:
            benchmark_options.extend(["--locale", locale])
        benchmark = get_benchmark(version, locale)
        with unittest.mock.patch("modelbench.run.find_suts_for_sut_argument") as mock_find_suts:
            mock_find_suts.return_value = [SutDescription("fake")]
            result = runner.invoke(
                cli,
                [
                    "benchmark",
                    "-m",
                    "1",
                    "--sut",
                    "fake",
                    "--output-dir",
                    str(tmp_path.absolute()),
                    *benchmark_options,
                ],
                catch_exceptions=False,
            )
            assert result.exit_code == 0
            assert (tmp_path / f"benchmark_record-{benchmark.uid}.json").exists

    def test_benchmark_anonymous_run_produces_json(self, runner, tmp_path):
        with unittest.mock.patch("modelbench.run.find_suts_for_sut_argument") as mock_find_suts:
            mock_find_suts.return_value = [SutDescription("fake")]
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

    def test_nonexistent_benchmark_versions_can_not_be_called(self, runner):
        result = runner.invoke(cli, ["benchmark", "--version", "0.0"])
        assert result.exit_code == 2
        assert "Invalid value for '--version'" in result.output

    def test_calls_score_benchmark_with_correct_v1_locale(self, runner, mock_score_benchmarks):
        result = runner.invoke(cli, ["benchmark", "--locale", "fr_FR"])

        benchmark_arg = mock_score_benchmarks.call_args.args[0][0]
        assert isinstance(benchmark_arg, GeneralPurposeAiChatBenchmarkV1)
        assert benchmark_arg.locale == Locale.FR_FR

    def test_calls_score_benchmark_all_locales(self, runner, mock_score_benchmarks, tmp_path):
        result = runner.invoke(cli, ["benchmark", "--locale", "all", "--output-dir", str(tmp_path.absolute())])

        benchmark_args = mock_score_benchmarks.call_args.args[0]
        locales = set([benchmark_arg.locale for benchmark_arg in benchmark_args])

        assert locales == {Locale.EN_US, Locale.FR_FR, Locale.HI_IN, Locale.ZH_CN}
        for locale in Locale:
            assert (tmp_path / f"benchmark_record-{GeneralPurposeAiChatBenchmarkV1(locale).uid}.json").exists

    def test_calls_score_benchmark_with_correct_version(self, runner, mock_score_benchmarks):
        result = runner.invoke(cli, ["benchmark", "--version", "0.5"])

        benchmark_arg = mock_score_benchmarks.call_args.args[0][0]
        assert isinstance(benchmark_arg, GeneralPurposeAiChatBenchmark)

    def test_v1_en_us_is_default(self, runner, mock_score_benchmarks):
        result = runner.invoke(cli, ["benchmark"])

        benchmark_arg = mock_score_benchmarks.call_args.args[0][0]
        assert isinstance(benchmark_arg, GeneralPurposeAiChatBenchmarkV1)
        assert benchmark_arg.locale == Locale.EN_US

    def test_warning_for_locale_option_with_version_0_5(self, runner, mock_score_benchmarks):
        with pytest.warns(UserWarning, match="Locale is not used in v0.5 benchmarks."):
            result = runner.invoke(cli, ["benchmark", "--locale", "en_US", "--version", "0.5"])

        benchmark_arg = mock_score_benchmarks.call_args.args[0][0]
        assert isinstance(benchmark_arg, GeneralPurposeAiChatBenchmark)
