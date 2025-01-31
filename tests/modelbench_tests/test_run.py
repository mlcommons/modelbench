import math
import unittest.mock
from datetime import datetime
from typing import List, Mapping, Sequence
from unittest.mock import MagicMock

import click
import pytest
from click.testing import CliRunner

from modelbench.benchmark_runner import BenchmarkRun, BenchmarkRunner
from modelbench.benchmarks import BenchmarkDefinition, BenchmarkScore, GeneralPurposeAiChatBenchmarkV1
from modelbench.hazards import HazardDefinition, HazardScore, SafeHazardV1
from modelbench.run import benchmark, cli, get_suts, get_benchmark
from modelbench.scoring import ValueEstimate
from modelgauge.base_test import PromptResponseTest
from modelgauge.locales import DEFAULT_LOCALE, EN_US, FR_FR, LOCALES
from modelgauge.prompt_sets import PROMPT_SETS
from modelgauge.records import TestRecord
from modelgauge.secret_values import RawSecrets
from modelgauge.sut import PromptResponseSUT

from modelgauge_tests.fake_sut import FakeSUT


class AHazard(HazardDefinition):
    def tests(self, secrets: RawSecrets) -> List[PromptResponseTest]:
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
    found_sut = get_suts([sut.uid])[0]
    assert isinstance(found_sut, FakeSUT)

    with pytest.raises(KeyError):
        get_suts(["something nonexistent"])


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

    @pytest.fixture(autouse=False)
    def mock_score_benchmarks(self, sut, monkeypatch):
        import modelbench

        mock = MagicMock(return_value=[self.mock_score(sut)])
        monkeypatch.setattr(modelbench.run, "score_benchmarks", mock)
        return mock

    @pytest.fixture(autouse=True)
    def do_print_summary(self, monkeypatch):
        import modelbench

        monkeypatch.setattr(modelbench.run, "print_summary", MagicMock())

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
        # TODO reenable when we re-add more languages:
        #  "version,locale", [("0.5", None), ("1.0", "en_US"), ("1.0", "fr_FR"), ("1.0", "hi_IN"), ("1.0", "zh_CN")]
    )
    def test_benchmark_basic_run_produces_json(
        self, runner, mock_score_benchmarks, sut_uid, version, locale, prompt_set, tmp_path
    ):
        benchmark_options = ["--version", version]
        if locale is not None:
            benchmark_options.extend(["--locale", locale])
        if prompt_set is not None:
            benchmark_options.extend(["--prompt-set", prompt_set])
        benchmark = get_benchmark(
            version,
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

    @pytest.mark.parametrize(
        "version,locale,prompt_set",
        [("1.0", None, None), ("1.0", EN_US, None), ("1.0", EN_US, "official")],
        # TODO: reenable when we re-add more languages
        # [("0.5", None), ("1.0", EN_US), ("1.0", FR_FR), ("1.0", HI_IN), ("1.0", ZH_CN)],
    )
    def test_benchmark_multiple_suts_produces_json(
        self, runner, version, locale, prompt_set, sut_uid, tmp_path, monkeypatch
    ):
        import modelbench

        benchmark_options = ["--version", version]
        if locale is not None:
            benchmark_options.extend(["--locale", locale])
        if prompt_set is not None:
            benchmark_options.extend(["--prompt-set", prompt_set])
        benchmark = get_benchmark(
            version,
            locale if locale else DEFAULT_LOCALE,
            prompt_set if prompt_set else "practice",
            "default",
        )

        mock = MagicMock(return_value=[self.mock_score(sut_uid, benchmark), self.mock_score("demo_yes_no", benchmark)])
        monkeypatch.setattr(modelbench.run, "score_benchmarks", mock)

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

    def test_benchmark_anonymous_run_produces_json(self, runner, sut_uid, tmp_path, mock_score_benchmarks):
        result = runner.invoke(
            cli,
            [
                "benchmark",
                "--anonymize",
                "42",
                "-m",
                "1",
                "--sut",
                sut_uid,
                "--output-dir",
                str(tmp_path.absolute()),
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.stdout
        assert (tmp_path / f"benchmark_record-{GeneralPurposeAiChatBenchmarkV1(EN_US, 'practice').uid}.json").exists

    @pytest.mark.parametrize("version", ["0.0", "0.5"])
    def test_invalid_benchmark_versions_can_not_be_called(self, version, runner):
        result = runner.invoke(cli, ["benchmark", "--version", "0.0"])
        assert result.exit_code == 2
        assert "Invalid value for '--version'" in result.output

    @pytest.mark.skip(reason="we have temporarily removed other languages")
    def test_calls_score_benchmark_with_correct_v1_locale(self, runner, mock_score_benchmarks, sut_uid):
        result = runner.invoke(cli, ["benchmark", "--locale", FR_FR, "--sut", sut_uid])

        benchmark_arg = mock_score_benchmarks.call_args.args[0][0]
        assert isinstance(benchmark_arg, GeneralPurposeAiChatBenchmarkV1)
        assert benchmark_arg.locale == FR_FR

    @pytest.mark.skip(reason="we have temporarily removed other languages")
    def test_calls_score_benchmark_all_locales(self, runner, mock_score_benchmarks, sut_uid, tmp_path):
        result = runner.invoke(
            cli, ["benchmark", "--locale", "all", "--output-dir", str(tmp_path.absolute()), "--sut", sut_uid]
        )

        benchmark_args = mock_score_benchmarks.call_args.args[0]
        locales = set([benchmark_arg.locale for benchmark_arg in benchmark_args])

        assert locales == LOCALES
        for locale in LOCALES:
            benchmark = GeneralPurposeAiChatBenchmarkV1(locale, "practice")
            assert (tmp_path / f"benchmark_record-{benchmark.uid}.json").exists

    # TODO: Add back when we add new versions.
    # def test_calls_score_benchmark_with_correct_version(self, runner, mock_score_benchmarks):
    #     result = runner.invoke(cli, ["benchmark", "--version", "0.5"])
    #
    #     benchmark_arg = mock_score_benchmarks.call_args.args[0][0]
    #     assert isinstance(benchmark_arg, GeneralPurposeAiChatBenchmark)

    def test_v1_en_us_demo_is_default(self, runner, mock_score_benchmarks, sut_uid):
        result = runner.invoke(cli, ["benchmark", "--sut", sut_uid])

        benchmark_arg = mock_score_benchmarks.call_args.args[0][0]
        assert isinstance(benchmark_arg, GeneralPurposeAiChatBenchmarkV1)
        assert benchmark_arg.locale == EN_US
        assert benchmark_arg.prompt_set == "demo"

    def test_nonexistent_benchmark_prompt_sets_can_not_be_called(self, runner, sut_uid):
        result = runner.invoke(cli, ["benchmark", "--prompt-set", "fake", "--sut", sut_uid])
        assert result.exit_code == 2
        assert "Invalid value for '--prompt-set'" in result.output

    def test_nonexistent_sut_uid_raises_exception(self, runner):
        result = runner.invoke(cli, ["benchmark", "--sut", "unknown-uid"])
        assert result.exit_code == 2
        assert "Invalid value for '--sut' / '-s': Unknown uid: '['unknown-uid']'" in result.output

    def test_multiple_nonexistent_sut_uids_raises_exception(self, runner):
        result = runner.invoke(cli, ["benchmark", "--sut", "unknown-uid1", "--sut", "unknown-uid2"])
        assert result.exit_code == 2
        assert "Invalid value for '--sut' / '-s': Unknown uids: '['unknown-uid1', 'unknown-uid2']'" in result.output

    @pytest.mark.parametrize("prompt_set", PROMPT_SETS.keys())
    def test_calls_score_benchmark_with_correct_prompt_set(self, runner, mock_score_benchmarks, prompt_set, sut_uid):
        result = runner.invoke(cli, ["benchmark", "--prompt-set", prompt_set, "--sut", sut_uid])

        benchmark_arg = mock_score_benchmarks.call_args.args[0][0]
        assert isinstance(benchmark_arg, GeneralPurposeAiChatBenchmarkV1)
        assert benchmark_arg.prompt_set == prompt_set
