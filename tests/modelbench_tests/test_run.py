import json
import math
from datetime import datetime
from typing import List, Mapping, Sequence
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from modelgauge_tests.fake_sut import FakeSUT

import modelbench
from modelbench.benchmark_runner import BenchmarkRun, BenchmarkRunner
from modelbench.benchmark_runner_items import ModelgaugeTestWrapper, TestRunItem
from modelbench.benchmarks import (
    BenchmarkDefinition,
    BenchmarkScore,
    GeneralPurposeAiChatBenchmarkV1,
    NaiveBenchmark,
    SecurityBenchmark,
    SecurityScore,
)
from modelbench.cli import cli
from modelbench.hazards import HazardDefinition, HazardScore, SafeHazardV1, Standards
from modelbench.scoring import ValueEstimate
from modelbench.standards import NoStandardsFileError, OverwriteStandardsFileError
from modelgauge.annotation import SafetyAnnotation
from modelgauge.base_test import PromptResponseTest
from modelgauge.dynamic_sut_factory import (
    ModelNotSupportedError,
    ProviderNotFoundError,
    UnknownSUTMakerError,
)
from modelgauge.locales import DEFAULT_LOCALE, EN_US, FR_FR, ZH_CN
from modelgauge.preflight import make_sut
from modelgauge.prompt import TextPrompt
from modelgauge.prompt_sets import GENERAL_PROMPT_SETS
from modelgauge.records import TestRecord
from modelgauge.secret_values import RawSecrets
from modelgauge.single_turn_prompt_response import TestItem
from modelgauge.sut import PromptResponseSUT, SUTResponse


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


def fake_benchmark_run(benchmark, hazards, sut, tmp_path):
    if isinstance(hazards, HazardDefinition):
        hazards = [hazards]
    benchmark_run = BenchmarkRun(BenchmarkRunner(tmp_path))
    benchmark_run.benchmarks = [benchmark]
    if isinstance(benchmark, SecurityBenchmark):
        score_cls = SecurityScore
    else:
        score_cls = BenchmarkScore
    benchmark_run.benchmark_scores[benchmark][sut] = score_cls(benchmark, sut, [h.score({}) for h in hazards], None)

    # these will be used to validate the contents of artifacts created by a run object
    test1 = ModelgaugeTestWrapper(
        actual_test=MagicMock(uid="test1-hazard-locale-official-1.1"), dependency_data_path=tmp_path
    )
    test_item1 = TestItem(
        prompt=TextPrompt(text="Why did the chicken cross the road?"), source_id="id1", shareable=True
    )
    tri1 = TestRunItem(
        test=test1,
        test_item=test_item1,
        sut=sut,
        sut_response=SUTResponse(text="To get to the other side"),
        annotations={"annotator1": SafetyAnnotation(is_safe=True)},
        measurements={},
        exceptions=[],
    )
    benchmark_run.add_finished_item(tri1)

    # this one should not be included in the annotations file b/c it's marked as non shareable
    test2 = ModelgaugeTestWrapper(
        actual_test=MagicMock(uid="test2-hazard-locale-practice-1.1"), dependency_data_path=tmp_path
    )
    test_item2 = TestItem(
        prompt=TextPrompt(text="What is the airspeed of an unladen swallow?"), source_id="id2", shareable=False
    )
    tri2 = TestRunItem(
        test=test2,
        test_item=test_item2,
        sut=sut,
        sut_response=SUTResponse(text="I don't know that"),
        annotations={"annotator2": SafetyAnnotation(is_safe=False)},
        measurements={},
        exceptions=[],
    )
    benchmark_run.add_finished_item(tri2)

    test3 = ModelgaugeTestWrapper(
        actual_test=MagicMock(uid="test3-hazard-locale-demo-1.1"), dependency_data_path=tmp_path
    )
    test_item3 = TestItem(prompt=TextPrompt(text="Who's a good boy?"), source_id="id3", shareable=True)
    tri3 = TestRunItem(
        test=test3,
        test_item=test_item3,
        sut=sut,
        sut_response=SUTResponse(text="Bark bork arf"),
        annotations={"annotator1": SafetyAnnotation(is_safe=True)},
        measurements={},
        exceptions=[],
    )
    benchmark_run.add_finished_item(tri3)

    # this one should not be included in the annotations file b/c it doesn't have availability set explicitly
    test4 = ModelgaugeTestWrapper(
        actual_test=MagicMock(uid="test4-hazard-locale-practice-1.1"), dependency_data_path=tmp_path
    )
    test_item4 = TestItem(
        prompt=TextPrompt(text="Is there anyone on board who knows how to fly a plane?"), source_id="id4"
    )
    tri4 = TestRunItem(
        test=test4,
        test_item=test_item4,
        sut=sut,
        sut_response=SUTResponse(text="Surely you can’t be serious."),
        annotations={"annotator2": SafetyAnnotation(is_safe=False)},
        measurements={},
        exceptions=[],
    )
    benchmark_run.add_finished_item(tri4)

    # this one should not be included in the annotations file b/c its prompt set is official
    test5 = ModelgaugeTestWrapper(
        actual_test=MagicMock(uid="test5-hazard-locale-official-1.1"), dependency_data_path=tmp_path
    )
    test_item5 = TestItem(
        prompt=TextPrompt(text="What is the way to Carnegie Hall?"),
        source_id="id5",
        shareable=True,
    )
    tri5 = TestRunItem(
        test=test5,
        test_item=test_item5,
        sut=sut,
        sut_response=SUTResponse(text="Practice, practice, practice."),
        annotations={"annotator1": SafetyAnnotation(is_safe=True)},
        measurements={},
        exceptions=[],
    )
    benchmark_run.add_finished_item(tri5)

    return benchmark_run


def annotations_are_correct(annotation_file_path, prompt_set) -> bool:
    assert annotation_file_path.exists()
    with open(annotation_file_path) as f:
        data = json.load(f)
    assert "job_id" in data
    assert "annotations" in data
    assert len(data["annotations"]) == 1
    assert data["annotations"][0]["response"] == "Bark bork arf"
    return True


def test_find_suts(sut):
    # key from modelbench gets a known SUT
    found_sut = make_sut(sut.uid)
    assert isinstance(found_sut, FakeSUT)

    with pytest.raises(ValueError):
        make_sut("something nonexistent")


@pytest.fixture
def standards_path_patch(monkeypatch, tmp_path):
    path = tmp_path / "standards.json"
    monkeypatch.setattr(
        Standards,
        "_benchmark_standards_path",
        classmethod(lambda cls, uid: path),
    )
    return path


@pytest.fixture(scope="module", autouse=True)
def fast_metadata():
    # Getting the benchmark metadata involves a lot of external processes, which slows our runs down quite a bit
    with mock.patch(
        "modelbench.record.benchmark_library_info", lambda: {"skipped by": "test_run.fast_metadata"}
    ) as _fixture:
        yield _fixture


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

    def mock_score_security(
        self,
        sut: PromptResponseSUT,
        benchmark=SecurityBenchmark(EN_US, "official"),
    ):
        est = ValueEstimate.make(0.123456, 100)
        return SecurityScore(
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
    def mock_run_benchmarks(self, sut, monkeypatch, tmp_path):
        hazards = [AHazard()]

        class ABenchmark(BenchmarkDefinition):
            def _make_hazards(self) -> Sequence[HazardDefinition]:
                return hazards

            @property
            def reference_suts(self) -> list[str]:
                return ["demo_yes_no"]

        benchmark = ABenchmark()
        mock = MagicMock(return_value=fake_benchmark_run(benchmark, hazards, sut, tmp_path))
        monkeypatch.setattr(modelbench.cli, "run_benchmarks_for_sut", mock)
        return mock

    @pytest.fixture(autouse=False)
    def mock_score_benchmarks(self, sut, monkeypatch):
        mock = MagicMock(return_value=[self.mock_score(sut)])
        monkeypatch.setattr(modelbench.cli, "score_benchmarks", mock)
        return mock

    @pytest.fixture(autouse=False)
    def mock_score_security_benchmarks(self, sut, monkeypatch):
        mock = MagicMock(return_value=[self.mock_score_security(sut)])
        monkeypatch.setattr(modelbench.cli, "score_benchmarks", mock)
        return mock

    @pytest.fixture(autouse=True)
    def do_print_summary(self, monkeypatch):
        monkeypatch.setattr(modelbench.cli, "print_summary", MagicMock())

    @pytest.fixture
    def run_dir(self, tmp_path):
        return tmp_path

    @pytest.fixture
    def runner(self, run_dir):
        runner = CliRunner()

        def invoke(command, args=None, **kwargs):
            args = list(args or [])
            full_args = ["--run-path", run_dir] + args
            return runner.invoke(command, full_args, **kwargs)

        return invoke

    @pytest.mark.parametrize(
        "version,locale,prompt_set",
        [
            ("1.1", None, None),
            ("1.1", EN_US, None),
            ("1.1", EN_US, "practice"),
            ("1.1", EN_US, "demo"),
            ("1.1", EN_US, "official"),
        ],
        # TODO add more locales as we add support for them
    )
    @pytest.mark.parametrize("sut_uid", ["fake-sut", "google/gemma-3-27b-it:scaleway:hfrelay"])
    def test_benchmark_basic_run_produces_json(
        self,
        runner,
        mock_run_benchmarks,
        mock_score_benchmarks,
        sut_uid,
        version,
        locale,
        prompt_set,
        run_dir,
    ):
        benchmark_options = ["--version", version]
        if locale is not None:
            benchmark_options.extend(["--locale", locale])
        if prompt_set is not None:
            benchmark_options.extend(["--prompt-set", prompt_set])
        benchmark = GeneralPurposeAiChatBenchmarkV1(
            locale if locale else DEFAULT_LOCALE,
            prompt_set if prompt_set else "demo",
            "default",
        )
        command_options = [
            "benchmark",
            "general",
            "-m",
            "1",
            "--sut",
            sut_uid,
            *benchmark_options,
        ]
        result = runner(
            cli,
            command_options,
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert (run_dir / "records" / f"benchmark_record-{benchmark.uid}.json").exists()

        annotation_file_path = run_dir / "records" / f"annotations-{benchmark.uid}.json"
        assert annotation_file_path.exists()
        # TODO find a better spot for this test. It's handy here because all the objects are available.
        assert annotations_are_correct(annotation_file_path, prompt_set)

    # TODO: Add test back after calibrating!!
    # def test_security_benchmark_basic_run_produces_json(
    #     self, runner, mock_run_benchmarks, mock_score_security_benchmarks, sut_uid, tmp_path
    # ):
    #     benchmark = SecurityBenchmark(EN_US, "official", "default")
    #     command_options = [
    #         "benchmark",
    #         "security",
    #         "-m",
    #         "1",
    #         "--sut",
    #         sut_uid,
    #         "--output-dir",
    #         str(tmp_path.absolute()),
    #     ]
    #     result = runner.invoke(
    #         cli,
    #         command_options,
    #         catch_exceptions=False,
    #     )
    #     assert result.exit_code == 0
    #     assert (tmp_path / f"benchmark_record-{benchmark.uid}.json").exists

    @pytest.mark.parametrize(
        "version,locale,prompt_set",
        [
            ("1.1", None, None),
            ("1.1", EN_US, None),
            ("1.1", EN_US, "official"),
            ("1.1", FR_FR, "practice"),
            ("1.1", FR_FR, "official"),
        ],
        # TODO add more locales as we add support for them
    )
    @pytest.mark.parametrize("sut_uid", ["fake-sut", "google/gemma-3-27b-it:scaleway:hfrelay;mt=500;t=0.3"])
    def test_benchmark_multiple_suts_produces_json(
        self, mock_run_benchmarks, runner, version, locale, prompt_set, sut_uid, run_dir, monkeypatch
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

        result = runner(
            cli,
            [
                "benchmark",
                "general",
                "-m",
                "1",
                "--sut",
                sut_uid,
                "--sut",
                "demo_yes_no",
                *benchmark_options,
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert (run_dir / "records" / f"benchmark_record-{benchmark.uid}.json").exists

    def test_benchmark_bad_sut_errors_out(self, runner):
        benchmark_options = ["--version", "1.1"]
        benchmark_options.extend(["--locale", "en_us"])
        benchmark_options.extend(["--prompt-set", "practice"])

        with pytest.raises(ValueError, match="No registration for bogus"):
            _ = runner(
                cli,
                [
                    "benchmark",
                    "general",
                    "-m",
                    "1",
                    "--sut",
                    "bogus",
                    *benchmark_options,
                ],
                catch_exceptions=False,
            )

        with pytest.raises(UnknownSUTMakerError):
            _ = runner(
                cli,
                [
                    "benchmark",
                    "general",
                    "-m",
                    "1",
                    "--sut",
                    "google/gemma:cohere:bogus",
                    *benchmark_options,
                ],
                catch_exceptions=False,
            )

        with patch(
            "modelgauge.suts.huggingface_sut_factory.HuggingFaceChatCompletionServerlessSUTFactory._find",
            side_effect=ProviderNotFoundError("bad provider"),
        ):
            with pytest.raises(ModelNotSupportedError):
                _ = runner(
                    cli,
                    [
                        "benchmark",
                        "general",
                        "-m",
                        "1",
                        "--sut",
                        "meta/llama:notreal:hfrelay",
                        *benchmark_options,
                    ],
                    catch_exceptions=False,
                )

        with patch(
            "modelgauge.suts.huggingface_sut_factory.hfh.model_info",
            side_effect=ModelNotSupportedError("bad model"),
        ):
            with pytest.raises(ModelNotSupportedError):
                _ = runner(
                    cli,
                    [
                        "benchmark",
                        "general",
                        "-m",
                        "1",
                        "--sut",
                        "google/bogus:cohere:hfrelay",
                        *benchmark_options,
                    ],
                    catch_exceptions=False,
                )

    @pytest.mark.parametrize("version", ["0.0", "0.5"])
    def test_invalid_benchmark_versions_can_not_be_called(self, version, runner):
        result = runner(cli, ["benchmark", "general", "--version", "0.0"])
        assert result.exit_code == 2
        assert "Invalid value for '--version'" in result.output

    @pytest.mark.skip(reason="we have temporarily removed other languages")
    def test_calls_score_benchmark_with_correct_v1_locale(self, runner, mock_run_benchmarks, sut_uid):
        _ = runner(cli, ["benchmark", "general", "--locale", FR_FR, "--sut", sut_uid])

        benchmark_arg = mock_run_benchmarks.call_args.args[0][0]
        assert isinstance(benchmark_arg, GeneralPurposeAiChatBenchmarkV1)
        assert benchmark_arg.locale == FR_FR

    # TODO: Add back when we add new versions.
    # def test_calls_score_benchmark_with_correct_version(self, runner, mock_score_benchmarks):
    #     result = runner(cli, ["benchmark", "general", "--version", "0.5"])
    #
    #     benchmark_arg = mock_score_benchmarks.call_args.args[0][0]
    #     assert isinstance(benchmark_arg, GeneralPurposeAiChatBenchmark)
    @pytest.mark.parametrize("sut_uid", ["fake-sut", "google/gemma-3-27b-it:scaleway:hfrelay"])
    def test_v1_en_us_demo_is_default(self, runner, mock_run_benchmarks, sut_uid):
        _ = runner(cli, ["benchmark", "general", "--sut", sut_uid])

        benchmark_arg = mock_run_benchmarks.call_args.args[0][0]
        assert isinstance(benchmark_arg, GeneralPurposeAiChatBenchmarkV1)
        assert benchmark_arg.locale == EN_US
        assert benchmark_arg.prompt_set == "demo"

    @pytest.mark.parametrize("sut_uid", ["fake-sut", "google/gemma-3-27b-it:scaleway:hfrelay"])
    def test_nonexistent_benchmark_prompt_sets_can_not_be_called(self, runner, sut_uid):
        result = runner(cli, ["benchmark", "general", "--prompt-set", "fake", "--sut", sut_uid])
        assert result.exit_code == 2
        assert "Invalid value for '--prompt-set'" in result.output

    @pytest.mark.parametrize("prompt_set", GENERAL_PROMPT_SETS.keys())
    @pytest.mark.parametrize("sut_uid", ["fake-sut", "google/gemma-3-27b-it:scaleway:hfrelay"])
    def test_calls_score_benchmark_with_correct_prompt_set(self, runner, mock_run_benchmarks, prompt_set, sut_uid):
        _ = runner(cli, ["benchmark", "general", "--prompt-set", prompt_set, "--sut", sut_uid])

        benchmark_arg = mock_run_benchmarks.call_args.args[0][0]
        assert isinstance(benchmark_arg, GeneralPurposeAiChatBenchmarkV1)
        assert benchmark_arg.prompt_set == prompt_set

    def test_fails_to_run_uncalibrated_benchmark(self, runner, mock_score_benchmarks, standards_path_patch):
        command_options = [
            "benchmark",
            "general",
            "-m",
            "1",
            "--sut",
            "fake-sut",
            "--locale",
            "fr_FR",
        ]
        with pytest.raises(NoStandardsFileError) as e:
            runner(
                cli,
                command_options,
                catch_exceptions=False,
            )
        assert e.value.path == standards_path_patch

    @pytest.mark.parametrize(
        "locale,prompt_set",
        [
            (EN_US, "practice"),
            (EN_US, "official"),
            (FR_FR, "practice"),
            (ZH_CN, "practice"),
        ],
    )
    def test_calibrate(
        self,
        runner,
        mock_score_benchmarks,
        sut_uid,
        sut,
        locale,
        prompt_set,
        standards_path_patch,
        tmp_path,
        monkeypatch,
    ):
        benchmark = GeneralPurposeAiChatBenchmarkV1(locale=locale, prompt_set=prompt_set)
        monkeypatch.setattr(GeneralPurposeAiChatBenchmarkV1, "reference_suts", [sut_uid])

        # Mock make_sut to return our fixture sut. This is so the cli can use it to key into the benchmark_scores.
        monkeypatch.setattr(modelbench.cli, "make_sut", lambda x: sut)

        # Mock run_benchmarks_for_sut
        reference_benchmark = benchmark.reference_benchmark()
        mock = MagicMock(
            return_value=fake_benchmark_run(reference_benchmark, reference_benchmark.hazards(), sut, tmp_path)
        )
        monkeypatch.setattr(modelbench.cli, "run_benchmarks_for_sut", mock)
        monkeypatch.setattr(modelbench.cli, "run_consistency_check", lambda *args, **kwargs: True)

        benchmark_options = []
        if locale is not None:
            benchmark_options.extend(["--locale", locale])
        if prompt_set is not None:
            benchmark_options.extend(["--prompt-set", prompt_set])

        command_options = [
            "calibrate",
            "general",
            "--evaluator",
            "default",
            *benchmark_options,
        ]

        result = runner(
            cli,
            command_options,
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert standards_path_patch.exists
        with open(standards_path_patch) as f:
            data = json.load(f)
        assert data["standards"]["reference_suts"] == [sut_uid]
        assert data["standards"]["reference_standards"] is not None

    def test_calibrate_security(
        self,
        runner,
        mock_score_security_benchmarks,
        sut_uid,
        sut,
        standards_path_patch,
        tmp_path,
        monkeypatch,
    ):
        locale = EN_US
        prompt_set = "official"

        benchmark = SecurityBenchmark(locale=locale, prompt_set=prompt_set)
        reference_benchmark = benchmark.reference_benchmark()
        monkeypatch.setattr(NaiveBenchmark, "reference_suts", [sut_uid])

        # Mock make_sut to return our fixture sut. This is so the cli can use it to key into the benchmark_scores.
        monkeypatch.setattr(modelbench.cli, "make_sut", lambda x: sut)

        # Mock run_benchmarks_for_sut
        mock = MagicMock(
            return_value=fake_benchmark_run(reference_benchmark, reference_benchmark.hazards(), sut, tmp_path)
        )
        monkeypatch.setattr(modelbench.cli, "run_benchmarks_for_sut", mock)
        monkeypatch.setattr(modelbench.cli, "run_consistency_check", lambda *args, **kwargs: True)

        benchmark_options = []
        if locale is not None:
            benchmark_options.extend(["--locale", locale])
        if prompt_set is not None:
            benchmark_options.extend(["--prompt-set", prompt_set])

        command_options = [
            "calibrate",
            "security",
            "--evaluator",
            "default",
            *benchmark_options,
        ]

        result = runner(
            cli,
            command_options,
            catch_exceptions=True,
        )
        assert result.exit_code == 0
        assert standards_path_patch.exists
        with open(standards_path_patch) as f:
            data = json.load(f)
        assert data["standards"]["reference_suts"] == [sut_uid]
        assert data["standards"]["reference_standards"] is not None

    def test_fails_to_calibrate_benchmark_with_standards(self, runner):
        command_options = [
            "calibrate",
            "general",
            "--locale",
            "en_us",
            "--prompt-set",
            "practice",
            "--evaluator",
            "default",
        ]
        with pytest.raises(OverwriteStandardsFileError) as e:
            runner(
                cli,
                command_options,
                catch_exceptions=False,
            )
