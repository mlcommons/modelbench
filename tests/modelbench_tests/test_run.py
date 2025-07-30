import functools
import math
import os
import pathlib
from datetime import datetime
from typing import List, Mapping, Sequence
from unittest.mock import MagicMock, patch

import modelbench

import pytest
from click.testing import CliRunner
from modelbench import hazards
from modelbench.benchmark_runner import BenchmarkRun, BenchmarkRunner
from modelbench.benchmarks import (
    BenchmarkDefinition,
    BenchmarkScore,
    GeneralPurposeAiChatBenchmarkV1,
    SecurityBenchmark,
)
from modelbench.hazards import HazardDefinition, HazardScore, SafeHazardV1, Standards
from modelbench.cli import benchmark, cli
from modelbench.scoring import ValueEstimate
from modelgauge.base_test import PromptResponseTest
from modelgauge.preflight import make_sut
from modelgauge.config import SECRETS_PATH
from modelgauge.dynamic_sut_factory import ModelNotSupportedError, ProviderNotFoundError, UnknownSUTMakerError
from modelgauge.locales import DEFAULT_LOCALE, EN_US, FR_FR, LOCALES
from modelgauge.prompt_sets import PROMPT_SETS
from modelgauge.records import TestRecord
from modelgauge.secret_values import RawSecrets
from modelgauge.sut import PromptResponseSUT
from modelgauge_tests.fake_sut import FakeSUT

TEST_SECRETS_PATH = os.path.join("tests", "config", "secrets.toml")


class AHazard(HazardDefinition):
    def tests(self, secrets: RawSecrets) -> List[PromptResponseTest]:
        pass

    def test_uids(self) -> str:
        pass

    def score(self, sut_scores: Mapping[str, TestRecord]) -> "HazardScore":
        est = ValueEstimate.make(0.123456, 100)
        return HazardScore(
            hazard_definition=self,
            score=est,
            test_scores={},
            exceptions=0,
            num_scored_items=6000,
            num_safe_items=math.floor(6000 * est.estimate),
        )


def fake_benchmark_run(hazards, sut, tmp_path):
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


def test_find_suts(sut):
    # key from modelbench gets a known SUT
    found_sut = make_sut(sut.uid)
    assert isinstance(found_sut, FakeSUT)

    with pytest.raises(ValueError):
        make_sut("something nonexistent")


class TestCli:
    class MyBenchmark(BenchmarkDefinition):
        def _make_hazards(self) -> Sequence[HazardDefinition]:
            return [SafeHazardV1(hazard, EN_US, "practice") for hazard in SafeHazardV1.all_hazard_keys]

        @property
        def uid(self):
            return "my_benchmark"

    def mock_score(
        self,
        sut: PromptResponseSUT,
        benchmark=GeneralPurposeAiChatBenchmarkV1(EN_US, "practice"),
    ):
        est = ValueEstimate.make(0.123456, 100)
        return BenchmarkScore(
            benchmark,
            sut,
            [
                HazardScore(
                    hazard_definition=benchmark.hazards()[0],
                    score=est,
                    test_scores={},
                    exceptions=0,
                    num_scored_items=10000,
                    num_safe_items=math.floor(10000 * est.estimate),
                ),
            ],
            datetime.now(),
        )

    def manage_test_secrets(func):
        """Decorator that manages test secrets during test execution.

        1. If a secrets file exists, it's backed up
        2. The test secrets file is copied to the expected location
        3. After the test completes, the original state is restored
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            secrets_src = pathlib.Path(TEST_SECRETS_PATH)
            secrets_dst = pathlib.Path(SECRETS_PATH)
            backup_dst = secrets_dst.with_suffix(".bak")

            if secrets_dst.exists():
                secrets_dst.replace(backup_dst)
            secrets_src.replace(secrets_dst)

            try:
                return func(*args, **kwargs)
            finally:
                secrets_dst.replace(secrets_src)
                if backup_dst.exists():
                    backup_dst.replace(secrets_dst)

        return wrapper

    @pytest.fixture(autouse=False)
    def mock_run_benchmarks(self, sut, monkeypatch, tmp_path):
        mock = MagicMock(return_value=fake_benchmark_run(AHazard(), sut, tmp_path))
        monkeypatch.setattr(modelbench.cli, "run_benchmarks_for_sut", mock)
        return mock

    @pytest.fixture(autouse=False)
    def mock_score_benchmarks(self, sut, monkeypatch):
        mock = MagicMock(return_value=[self.mock_score(sut)])
        monkeypatch.setattr(modelbench.cli, "score_benchmarks", mock)
        return mock

    @pytest.fixture(autouse=True)
    def do_print_summary(self, monkeypatch):
        monkeypatch.setattr(modelbench.cli, "print_summary", MagicMock())

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.mark.parametrize(
        "version,locale,prompt_set",
        [
            ("1.0", None, None),
            ("1.0", EN_US, None),
            ("1.0", EN_US, "practice"),
            ("1.0", EN_US, "official"),
        ],
        # TODO add more locales as we add support for them
    )
    @manage_test_secrets
    def test_benchmark_basic_run_produces_json(
        self, runner, mock_run_benchmarks, mock_score_benchmarks, sut_uid, version, locale, prompt_set, tmp_path
    ):
        benchmark_options = ["--version", version]
        if locale is not None:
            benchmark_options.extend(["--locale", locale])
        if prompt_set is not None:
            benchmark_options.extend(["--prompt-set", prompt_set])
        benchmark = GeneralPurposeAiChatBenchmarkV1(
            locale if locale else DEFAULT_LOCALE,
            prompt_set if prompt_set else "practice",
            "default",
        )
        command_options = [
            "benchmark",
            "-m",
            "1",
            "--sut",
            sut_uid,
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

    def test_security_benchmark_basic_run_produces_json(
        self, runner, mock_run_benchmarks, mock_score_benchmarks, sut_uid, tmp_path
    ):
        benchmark = SecurityBenchmark("default")
        command_options = [
            "security-benchmark",
            "-m",
            "1",
            "--sut",
            sut_uid,
            "--output-dir",
            str(tmp_path.absolute()),
        ]
        result = runner.invoke(
            cli,
            command_options,
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert (tmp_path / f"benchmark_record-{benchmark.uid}.json").exists

    @pytest.mark.parametrize(
        "version,locale,prompt_set",
        [
            ("1.0", None, None),
            ("1.0", EN_US, None),
            ("1.0", EN_US, "official"),
            ("1.0", FR_FR, "practice"),
            ("1.0", FR_FR, "official"),
        ],
        # TODO add more locales as we add support for them
    )
    @manage_test_secrets
    def test_benchmark_multiple_suts_produces_json(
        self, mock_run_benchmarks, runner, version, locale, prompt_set, sut_uid, tmp_path, monkeypatch
    ):

        benchmark_options = ["--version", version]
        if locale is not None:
            benchmark_options.extend(["--locale", locale])
        if prompt_set is not None:
            benchmark_options.extend(["--prompt-set", prompt_set])
        benchmark = GeneralPurposeAiChatBenchmarkV1(
            locale if locale else DEFAULT_LOCALE,
            prompt_set if prompt_set else "practice",
            "default",
        )

        mock = MagicMock(return_value=[self.mock_score(sut_uid, benchmark), self.mock_score("demo_yes_no", benchmark)])
        monkeypatch.setattr(modelbench.cli, "score_benchmarks", mock)

        result = runner.invoke(
            cli,
            [
                "benchmark",
                "-m",
                "1",
                "--sut",
                sut_uid,
                "--sut",
                "demo_yes_no",
                "--output-dir",
                str(tmp_path.absolute()),
                *benchmark_options,
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert (tmp_path / f"benchmark_record-{benchmark.uid}.json").exists

    def test_benchmark_bad_sut_errors_out(self, runner, tmp_path):
        benchmark_options = ["--version", "1.0"]
        benchmark_options.extend(["--locale", "en_us"])
        benchmark_options.extend(["--prompt-set", "practice"])

        with pytest.raises(ValueError, match="No registration for bogus"):
            _ = runner.invoke(
                cli,
                [
                    "benchmark",
                    "-m",
                    "1",
                    "--sut",
                    "bogus",
                    "--output-dir",
                    str(tmp_path.absolute()),
                    *benchmark_options,
                ],
                catch_exceptions=False,
            )

        with pytest.raises(UnknownSUTMakerError):
            _ = runner.invoke(
                cli,
                [
                    "benchmark",
                    "-m",
                    "1",
                    "--sut",
                    "google/gemma:cohere:bogus",
                    "--output-dir",
                    str(tmp_path.absolute()),
                    *benchmark_options,
                ],
                catch_exceptions=False,
            )

        with patch(
            "modelgauge.suts.huggingface_sut_factory.HuggingFaceChatCompletionServerlessSUTFactory._find",
            side_effect=ProviderNotFoundError("bad provider"),
        ):
            with pytest.raises(ProviderNotFoundError):
                _ = runner.invoke(
                    cli,
                    [
                        "benchmark",
                        "-m",
                        "1",
                        "--sut",
                        "google/gemma:bogus:hfrelay",
                        "--output-dir",
                        str(tmp_path.absolute()),
                        *benchmark_options,
                    ],
                    catch_exceptions=False,
                )

        with patch(
            "modelgauge.suts.huggingface_sut_factory.hfh.model_info",
            side_effect=ModelNotSupportedError("bad model"),
        ):
            with pytest.raises(ModelNotSupportedError):
                _ = runner.invoke(
                    cli,
                    [
                        "benchmark",
                        "-m",
                        "1",
                        "--sut",
                        "google/bogus:cohere:hfrelay",
                        "--output-dir",
                        str(tmp_path.absolute()),
                        *benchmark_options,
                    ],
                    catch_exceptions=False,
                )

    @pytest.mark.parametrize("version", ["0.0", "0.5"])
    def test_invalid_benchmark_versions_can_not_be_called(self, version, runner):
        result = runner.invoke(cli, ["benchmark", "--version", "0.0"])
        assert result.exit_code == 2
        assert "Invalid value for '--version'" in result.output

    @pytest.mark.skip(reason="we have temporarily removed other languages")
    def test_calls_score_benchmark_with_correct_v1_locale(self, runner, mock_run_benchmarks, sut_uid):
        result = runner.invoke(cli, ["benchmark", "--locale", FR_FR, "--sut", sut_uid])

        benchmark_arg = mock_run_benchmarks.call_args.args[0][0]
        assert isinstance(benchmark_arg, GeneralPurposeAiChatBenchmarkV1)
        assert benchmark_arg.locale == FR_FR

    # TODO: Add back when we add new versions.
    # def test_calls_score_benchmark_with_correct_version(self, runner, mock_score_benchmarks):
    #     result = runner.invoke(cli, ["benchmark", "--version", "0.5"])
    #
    #     benchmark_arg = mock_score_benchmarks.call_args.args[0][0]
    #     assert isinstance(benchmark_arg, GeneralPurposeAiChatBenchmark)

    @manage_test_secrets
    def test_v1_en_us_demo_is_default(self, runner, mock_run_benchmarks, sut_uid):
        result = runner.invoke(cli, ["benchmark", "--sut", sut_uid])

        benchmark_arg = mock_run_benchmarks.call_args.args[0][0]
        assert isinstance(benchmark_arg, GeneralPurposeAiChatBenchmarkV1)
        assert benchmark_arg.locale == EN_US
        assert benchmark_arg.prompt_set == "demo"

    def test_nonexistent_benchmark_prompt_sets_can_not_be_called(self, runner, sut_uid):
        result = runner.invoke(cli, ["benchmark", "--prompt-set", "fake", "--sut", sut_uid])
        assert result.exit_code == 2
        assert "Invalid value for '--prompt-set'" in result.output

    @pytest.mark.parametrize("prompt_set", PROMPT_SETS.keys())
    @manage_test_secrets
    def test_calls_score_benchmark_with_correct_prompt_set(self, runner, mock_run_benchmarks, prompt_set, sut_uid):
        result = runner.invoke(cli, ["benchmark", "--prompt-set", prompt_set, "--sut", sut_uid])

        benchmark_arg = mock_run_benchmarks.call_args.args[0][0]
        assert isinstance(benchmark_arg, GeneralPurposeAiChatBenchmarkV1)
        assert benchmark_arg.prompt_set == prompt_set

    def test_fails(self, runner, mock_score_benchmarks, sut_uid, tmp_path, monkeypatch):
        standards = Standards(pathlib.Path(__file__).parent / "data" / "standards_with_en_us_practice_only.json")

        monkeypatch.setattr(hazards, "STANDARDS", standards)

        command_options = [
            "benchmark",
            "-m",
            "1",
            "--sut",
            sut_uid,
            "--output-dir",
            str(tmp_path.absolute()),
            "--locale",
            "fr_FR",
        ]
        with pytest.raises(ValueError) as e:
            runner.invoke(
                cli,
                command_options,
                catch_exceptions=False,
            )
        assert "No standard yet for" in str(e.value)
