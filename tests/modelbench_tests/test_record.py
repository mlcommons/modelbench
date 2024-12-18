import json
import math
import platform
import pytest
import re
from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch

from modelbench.benchmarks import BenchmarkScore, GeneralPurposeAiChatBenchmarkV1
from modelbench.hazards import HazardScore, SafeHazardV1
from modelbench.record import (
    benchmark_code_info,
    benchmark_run_record,
    BenchmarkScoreEncoder,
    dump_json,
)
from modelbench.scoring import ValueEstimate

from modelgauge.record_init import InitializationRecord
from modelgauge.tests.safe_v1 import Locale


@pytest.fixture()
def benchmark_score(end_time, sut):
    bd = GeneralPurposeAiChatBenchmarkV1(Locale.EN_US, "practice")
    low_est = ValueEstimate.make(0.5, 10)
    high_est = ValueEstimate.make(0.8, 20)
    bs = BenchmarkScore(
        bd,
        sut,
        [
            HazardScore(
                hazard_definition=SafeHazardV1("cse", Locale.EN_US, "practice"),
                score=low_est,
                test_scores={},
                exceptions=0,
                num_scored_items=10000,
                num_safe_items=math.floor(low_est.estimate * 10000),
            ),
            HazardScore(
                hazard_definition=SafeHazardV1("dfm", Locale.EN_US, "practice"),
                score=high_est,
                test_scores={},
                exceptions=0,
                num_scored_items=10000,
                num_safe_items=math.floor(high_est.estimate * 10000),
            ),
        ],
        end_time=end_time,
    )
    return bs


def encode(o):
    return json.dumps(o, cls=BenchmarkScoreEncoder, indent=4)


def encode_and_parse(o):
    s = encode(o)
    return json.loads(s)


def test_sut(sut):
    encoded = encode_and_parse(sut)
    assert encoded["uid"] == sut.uid
    assert "initialization" in encoded


def test_value_estimate():
    ve = ValueEstimate.make(0.5, 1000)
    j = encode_and_parse(ve)
    assert j["lower"] == ve.lower
    assert j["estimate"] == ve.estimate
    assert j["upper"] == ve.upper
    assert j["samples"] == ve.samples


def test_v1_hazard_definition_without_tests_loaded():
    hazard = SafeHazardV1("dfm", Locale.EN_US, "practice")
    j = encode_and_parse(hazard)
    assert j["uid"] == hazard.uid
    assert "tests" not in j
    assert j["reference_standard"] == hazard.reference_standard()


def test_v1_hazard_definition_with_tests_loaded():
    hazard = SafeHazardV1("dfm", Locale.EN_US, "practice")
    hazard.tests({"together": {"api_key": "ignored"}, "modellab_files": {"token": "ignored"}})
    j = encode_and_parse(hazard)
    assert j["uid"] == hazard.uid
    assert j["tests"] == ["safe-dfm-en_us-practice-1.0"]
    assert j["reference_standard"] == hazard.reference_standard()


def test_benchmark_definition():
    j = encode_and_parse(GeneralPurposeAiChatBenchmarkV1(locale=Locale.EN_US, prompt_set="practice"))
    assert j["uid"] == "general_purpose_ai_chat_benchmark-1.0-en_us-practice-default"
    assert "safe_hazard-1.0-cse-en_us-practice" in [i["uid"] for i in j["hazards"]]


def test_hazard_score():
    hazard = SafeHazardV1("cse", Locale.EN_US, "practice")
    ve = ValueEstimate.make(1.0, 100000)
    hs = HazardScore(hazard_definition=hazard, score=ve, test_scores={"cse": ve}, exceptions=0)
    j = encode_and_parse(hs)
    assert j["hazard_definition"]["uid"] == hazard.uid
    assert j["score"]["samples"] == ve.samples
    assert j["test_scores"]["cse"]["samples"] == ve.samples
    assert j["numeric_grade"] == 5
    assert j["text_grade"] == "E"
    assert j["exceptions"] == 0


def test_benchmark_score(benchmark_score):
    j = encode_and_parse(benchmark_score)
    assert "benchmark_definition" not in j  # it's already higher up in the tree; no need to duplicate
    assert j["sut"]["uid"] == benchmark_score.sut.uid
    assert len(j["hazard_scores"]) == len(benchmark_score.hazard_scores)
    assert j["end_time"] == str(benchmark_score.end_time)
    assert j["numeric_grade"] == benchmark_score.numeric_grade()
    assert j["text_grade"] == benchmark_score.text_grade()


def test_benchmark_score_initialization_record(benchmark_score):
    benchmark_score.sut.initialization_record = InitializationRecord(
        module="a_module", class_name="a_class", args=["arg1", "arg2"], kwargs={"kwarg1": "a_value"}
    )
    j = encode_and_parse(benchmark_score)
    assert j["sut"]["initialization"]["module"] == "a_module"


def test_benchmark_run_record(benchmark_score):
    r = benchmark_run_record(benchmark_score)
    assert r["score"] == benchmark_score
    assert r["_metadata"]["format_version"] == 1

    run_info = r["_metadata"]["run"]
    assert re.match(r"\w+", run_info["user"])
    assert re.match(r"20\d\d-.+UTC", run_info["timestamp"])
    assert run_info["platform"] == platform.platform()
    assert run_info["system"]
    assert run_info["node"] == platform.node()
    assert run_info["python"] == platform.python_version()


def test_benchmark_code_record(benchmark_score):
    r = benchmark_run_record(benchmark_score)
    source = r["_metadata"]["code"]["source"]
    assert source["git_version"].startswith("git version 2")
    assert source["origin"] in [
        "git@github.com:mlcommons/modelbench.git",
        "https://github.com/mlcommons/modelbench",
        "https://github.com/mlcommons/modelbench.git",
        "git@github.com:mlcommons/modelbench-private.git",
        "https://github.com/mlcommons/modelbench-private",
        "https://github.com/mlcommons/modelbench-private.git",
    ]
    assert re.match(r"(v[.0-9]+-\d+-)?[a-z0-9]{8}", source["code_version"])
    assert isinstance(source["changed_files"], list)  # hard to be more specific here


def test_benchmark_code_record_without_git_command(benchmark_score):
    with patch("modelbench.record.run_command") as f:
        f.side_effect = FileNotFoundError()
        j = benchmark_code_info()
        print(j)
        assert j["error"].startswith("git command not found")


def test_benchmark_code_record_without_git_repo(benchmark_score, cwd_tmpdir):
    j = benchmark_code_info()
    print(j)
    assert j["error"].startswith("couldn't find git dir")


def test_benchmark_code_record_without_git(benchmark_score):
    with patch("modelbench.record.run_command") as f:
        f.side_effect = FileNotFoundError()
        r = benchmark_run_record(benchmark_score)
        source = r["_metadata"]["code"]["source"]
        assert source["error"] == "git command not found"


def test_dump_json(benchmark_score, tmp_path):
    # just a smoke test; everything substantial should be tested above.
    json_path = tmp_path / "foo.json"
    dump_json(
        json_path,
        datetime.fromtimestamp(1700000000, timezone.utc),
        benchmark_score.benchmark_definition,
        [benchmark_score],
    )
    with open(json_path) as f:
        j = json.load(f)
    assert "_metadata" in j
    assert j["benchmark"]["uid"] == benchmark_score.benchmark_definition.uid
    assert j["run_uid"] == "run-" + benchmark_score.benchmark_definition.uid + "-20231114-221320"
    assert "grades" in j["content"]
    assert len(j["scores"]) == 1
