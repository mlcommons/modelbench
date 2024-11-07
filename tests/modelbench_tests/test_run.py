import json
import pathlib
import unittest.mock
from datetime import datetime
from typing import Sequence, Mapping, List
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

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
from modelgauge.base_test import PromptResponseTest
from modelgauge.records import TestRecord
from modelgauge.secret_values import RawSecrets
from modelgauge.tests.safe_v1 import Locale


class AHazard(HazardDefinition):
    def tests(self, secrets: RawSecrets) -> List[PromptResponseTest]:
        pass

    def score(self, sut_scores: Mapping[str, TestRecord]) -> "HazardScore":
        return HazardScore(
            hazard_definition=self, score=ValueEstimate.make(0.123456, 100), test_scores={}, exceptions=0
        )


def fake_benchmark_run(hazards, tmp_path):
    sut = ModelGaugeSut.for_key("mistral-7b")
    if isinstance(hazards, HazardDefinition):
        hazards = [hazards]

    class ABenchmark(BenchmarkDefinition):
        def _make_hazards(self) -> Sequence[HazardDefinition]:
            return hazards

    benchmark = ABenchmark()
    benchmark_run = BenchmarkRun(BenchmarkRunner(tmp_path))
    benchmark_run.benchmarks = [benchmark]
    benchmark_run.benchmark_scores[benchmark][sut] = BenchmarkScore(
        benchmark, sut, [h.score({}) for h in hazards], None
    )
    return benchmark_run


def test_find_suts():
    # nothing gets everything
    assert find_suts_for_sut_argument([]) == SUTS_FOR_V_0_5

    # key from modelbench gets a known SUT
    assert find_suts_for_sut_argument(["mistral-7b"]) == [ModelGaugeSut.for_key("mistral-7b")]

    # key from modelgauge gets a dynamic one
    dynamic_qwen = find_suts_for_sut_argument(["llama-3-70b-chat-hf"])[0]
    assert dynamic_qwen.key == "llama-3-70b-chat-hf"

    with pytest.raises(click.BadParameter):
        find_suts_for_sut_argument(["something nonexistent"])


class TestCli:
    class MyBenchmark(BenchmarkDefinition):
        def _make_hazards(self) -> Sequence[HazardDefinition]:
            return [c() for c in SafeHazard.__subclasses__()]

        @property
        def uid(self):
            return "my_benchmark"

    def mock_score(self, benchmark=GeneralPurposeAiChatBenchmark(), sut=ModelGaugeSut.for_key("mistral-7b")):
        return BenchmarkScore(
            benchmark,
            sut,
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

    @pytest.fixture(autouse=False)
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
        "version,locale",
        [("0.5", None), ("1.0", None), ("1.0", "en_US")],
        # TODO reenable when we re-add more languages:
        #  "version,locale", [("0.5", None), ("1.0", "en_US"), ("1.0", "fr_FR"), ("1.0", "hi_IN"), ("1.0", "zh_CN")]
    )
    def test_benchmark_basic_run_produces_json(self, runner, mock_score_benchmarks, version, locale, tmp_path):
        benchmark_options = ["--version", version]
        if locale is not None:
            benchmark_options.extend(["--locale", locale])
        benchmark = get_benchmark(version, locale if locale else Locale.EN_US, "default")
        with unittest.mock.patch("modelbench.run.find_suts_for_sut_argument") as mock_find_suts:
            mock_find_suts.return_value = [SutDescription("fake")]
            command_options = [
                "benchmark",
                "-m",
                "1",
                "--sut",
                "fake",
                "--output-dir",
                str(tmp_path.absolute()),
                *benchmark_options,
            ]
            result = runner.invoke(
                cli,
                command_options,
                catch_exceptions=False,
            )
            assert result.exit_code == 0
            assert (tmp_path / f"benchmark_record-{benchmark.uid}.json").exists

    @pytest.mark.parametrize(
        "version,locale",
        [("0.5", None), ("0.5", None), ("1.0", Locale.EN_US)],
        # TODO: reenable when we re-add more languages
        # [("0.5", None), ("1.0", Locale.EN_US), ("1.0", Locale.FR_FR), ("1.0", Locale.HI_IN), ("1.0", Locale.ZH_CN)],
    )
    def test_benchmark_multiple_suts_produces_json(self, runner, version, locale, tmp_path, monkeypatch):
        import modelbench

        benchmark_options = ["--version", version]
        if locale is not None:
            benchmark_options.extend(["--locale", locale.value])
        benchmark = get_benchmark(version, locale, "default")

        mock = MagicMock(return_value=[self.mock_score(benchmark, "fake-2"), self.mock_score(benchmark, "fake-2")])
        monkeypatch.setattr(modelbench.run, "score_benchmarks", mock)
        with unittest.mock.patch("modelbench.run.find_suts_for_sut_argument") as mock_find_suts:
            mock_find_suts.return_value = [SutDescription("fake-1"), SutDescription("fake-2")]
            result = runner.invoke(
                cli,
                [
                    "benchmark",
                    "-m",
                    "1",
                    "--sut",
                    "fake-1",
                    "--sut",
                    "fake-2",
                    "--output-dir",
                    str(tmp_path.absolute()),
                    *benchmark_options,
                ],
                catch_exceptions=False,
            )
            assert result.exit_code == 0
            assert (tmp_path / f"benchmark_record-{benchmark.uid}.json").exists

    def test_benchmark_anonymous_run_produces_json(self, runner, tmp_path, mock_score_benchmarks):
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

    @pytest.mark.skip(reason="we have temporarily removed other languages")
    def test_calls_score_benchmark_with_correct_v1_locale(self, runner, mock_score_benchmarks):
        result = runner.invoke(cli, ["benchmark", "--locale", "fr_FR"])

        benchmark_arg = mock_score_benchmarks.call_args.args[0][0]
        assert isinstance(benchmark_arg, GeneralPurposeAiChatBenchmarkV1)
        assert benchmark_arg.locale == Locale.FR_FR

    @pytest.mark.skip(reason="we have temporarily removed other languages")
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
