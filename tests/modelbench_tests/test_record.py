import json
import math
import os
import platform
import re
from datetime import datetime, timezone
from unittest import mock
from unittest.mock import patch

import pytest

from modelbench.benchmarks import (
    BenchmarkScore,
    GeneralPurposeAiChatBenchmarkV1,
    SecurityBenchmark,
    SecurityScore,
)
from modelbench.hazards import HazardScore, SafeHazardV1, SecurityJailbreakHazard
from modelbench.record import BenchmarkScoreEncoder, benchmark_code_info, dump_json
from modelbench.scoring import ValueEstimate
from modelbench.standards import Standards
from modelgauge.locales import EN_US
from modelgauge.record_init import InitializationRecord
from modelgauge.sut import PromptResponseSUT
from modelgauge.sut_decorator import modelgauge_sut


def benchmark_run_record(benchmark_score):
    # this function is slow for reasons we generally don't care about, so fake the slow part
    from modelbench.record import benchmark_run_record as real_benchmark_run_record

    with mock.patch("modelbench.record.benchmark_library_info", lambda: {"skipped by": "test_run.fast_metadata"}):
        return real_benchmark_run_record(benchmark_score)


@pytest.fixture()
def secrets():
    return {"together": {"api_key": "fake"}, "modellab_files": {"token": "fake"}}


def benchmark_score_gen(end_time, sut):
    bd = GeneralPurposeAiChatBenchmarkV1(EN_US, "practice")
    low_est = ValueEstimate.make(0.5, 10)
    high_est = ValueEstimate.make(0.8, 20)
    bs = BenchmarkScore(
        bd,
        sut,
        [
            HazardScore(
                hazard_definition=bd.hazards()[0],
                score=low_est,
                test_scores={},
                exceptions=0,
                num_scored_items=10000,
                num_safe_items=math.floor(low_est.estimate * 10000),
            ),
            HazardScore(
                hazard_definition=bd.hazards()[1],
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


@pytest.fixture()
def benchmark_score(end_time, sut):
    return benchmark_score_gen(end_time, sut)


@pytest.fixture()
def unserializable_sut():
    @modelgauge_sut(capabilities=[])
    class UnserializableSUT(PromptResponseSUT):
        def __init__(self, uid, unserializable_init_param):
            super().__init__(uid)

        def evaluate(self, request):
            pass

        def translate_response(self, request, response):
            pass

    return UnserializableSUT("unserializable_sut", unserializable_init_param=lambda x: x)


@pytest.fixture()
def benchmark_score_with_unserializable_sut(end_time, unserializable_sut):
    return benchmark_score_gen(end_time, unserializable_sut)


@pytest.fixture()
def security_score(monkeypatch, tmp_path, end_time, sut):
    # Patch the standards in case the benchmark is not yet calibrated.
    file = tmp_path / "benchmark_standards.json"
    hazard = SecurityJailbreakHazard(EN_US, "official")
    standards = {
        "reference_standards": {hazard.reference_key: 0.8},
    }
    with open(file, "w") as out:
        json.dump({"standards": standards, "_metadata": {"run_info": {}}}, out)
    monkeypatch.setattr(
        Standards,
        "_benchmark_standards_path",
        classmethod(lambda cls, uid: file),
    )

    bd = SecurityBenchmark(EN_US, "official")
    low_est = ValueEstimate.make(0.5, 10)
    high_est = ValueEstimate.make(0.8, 20)
    bs = SecurityScore(
        bd,
        sut,
        [
            HazardScore(
                hazard_definition=bd.hazards()[0],
                score=low_est,
                test_scores={},
                exceptions=0,
                num_scored_items=10000,
                num_safe_items=math.floor(low_est.estimate * 10000),
            ),
            HazardScore(
                hazard_definition=bd.hazards()[1],
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


def test_unserializable_sut_within_benchmark_score(benchmark_score_with_unserializable_sut):
    encoded = encode(benchmark_score_with_unserializable_sut)
    assert '"unserializable_init_param": "Object of type function is not JSON serializable"' in encoded


def test_value_estimate():
    ve = ValueEstimate.make(0.5, 1000)
    j = encode_and_parse(ve)
    assert j["lower"] == ve.lower
    assert j["estimate"] == ve.estimate
    assert j["upper"] == ve.upper
    assert j["samples"] == ve.samples


def test_v1_hazard_definition_without_tests_loaded():
    hazard = SafeHazardV1("dfm", EN_US, "practice")
    j = encode_and_parse(hazard)
    assert j["uid"] == hazard.uid
    assert "tests" not in j
    assert j["reference_standard"] == hazard.reference_standard()


def test_v1_hazard_definition_with_tests_loaded(secrets):
    hazard = SafeHazardV1("dfm", EN_US, "practice")
    hazard.tests(secrets)
    j = encode_and_parse(hazard)
    assert j["uid"] == hazard.uid
    assert j["tests"] == ["safe-dfm-en_us-practice-1.1"]
    assert j["reference_standard"] == hazard.reference_standard()


def test_general_benchmark_definition():
    j = encode_and_parse(GeneralPurposeAiChatBenchmarkV1(locale=EN_US, prompt_set="practice"))
    assert j["uid"] == "general_purpose_ai_chat_benchmark-1.1-en_us-practice-default"
    assert j["version"] == "1.1"
    assert j["prompt_set"] == "practice"
    assert "safe_hazard-1.1-cse-en_us-practice" in [i["uid"] for i in j["hazards"]]


def test_security_benchmark_definition():
    j = encode_and_parse(SecurityBenchmark(locale=EN_US, prompt_set="official"))
    assert j["uid"] == "security_benchmark-0.5-en_us-official-default"
    assert j["version"] == "0.5"
    hazard_uids = [i["uid"] for i in j["hazards"]]
    assert "security_jailbreak_hazard-0.5-en_us-official" in hazard_uids


def test_hazard_score():
    hazard = SafeHazardV1("cse", EN_US, "practice")
    hazard.set_standard(GeneralPurposeAiChatBenchmarkV1(locale=EN_US, prompt_set="practice").standards)
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


def test_security_score(security_score):
    j = encode_and_parse(security_score)
    assert "benchmark_definition" not in j  # it's already higher up in the tree; no need to duplicate
    assert j["sut"]["uid"] == security_score.sut.uid
    assert len(j["hazard_scores"]) == len(security_score.hazard_scores)
    assert j["end_time"] == str(security_score.end_time)
    assert j["numeric_grade"] is None
    assert j["text_grade"] == "N/A"
    assert j["score"] is None  # Needs to be None! Not 0.0


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


@pytest.mark.parametrize("run_uid", [None, "custom_run_uid"])
def test_dump_json(benchmark_score, tmp_path, run_uid):
    # just a smoke test; everything substantial should be tested above.
    json_path = tmp_path / "foo.json"
    with mock.patch("modelbench.record.benchmark_library_info", lambda: {"skipped by": "test_run.fast_metadata"}):
        dump_json(
            json_path,
            datetime.fromtimestamp(1700000000, timezone.utc),
            benchmark_score.benchmark_definition,
            [benchmark_score],
            run_uid,
        )

    with open(json_path) as f:
        j = json.load(f)
    assert "_metadata" in j
    assert j["benchmark"]["uid"] == benchmark_score.benchmark_definition.uid
    if not run_uid:
        assert j["run_uid"] == "run-" + benchmark_score.benchmark_definition.uid + "-20231114-221320"
    else:
        assert j["run_uid"] == run_uid
    assert len(j["scores"]) == 1


def test_dump_json_user(benchmark_score, tmp_path):
    print("TESTING")

    def dump_and_read_record(**kwargs):
        json_path = tmp_path / "foo.json"
        with mock.patch("modelbench.record.benchmark_library_info", lambda: {"skipped by": "test_run.fast_metadata"}):
            dump_json(
                json_path,
                datetime.fromtimestamp(1700000000, timezone.utc),
                benchmark_score.benchmark_definition,
                [benchmark_score],
                None,
                **kwargs,
            )
        with open(json_path) as f:
            data = json.load(f)
        return data

    saved_user = os.environ.get("USER")
    os.environ["USER"] = "me"
    try:
        # Dump correctly writes user from environment variable
        j = dump_and_read_record()
        assert j["_metadata"]["run"]["user"] == "me"

        # Dump correctly writes user from argument
        j = dump_and_read_record(user="custom_user")
        assert j["_metadata"]["run"]["user"] == "custom_user"
    finally:
        # Restore environment
        if saved_user is None:
            del os.environ["USER"]
        else:
            os.environ["USER"] = saved_user
