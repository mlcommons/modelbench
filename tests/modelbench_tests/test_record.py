import json
import platform
import re
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock, patch

from modelgauge.record_init import InitializationRecord

from modelbench.benchmarks import GeneralPurposeAiChatBenchmark
from modelbench.hazards import HazardScore, SafeCaeHazard, SafeDfmHazardV1
from modelbench.record import (
    BenchmarkScoreEncoder,
    benchmark_run_record,
    dump_json,
    benchmark_code_info,
    benchmark_library_info,
)
from modelbench.run import FakeSut
from modelbench.scoring import ValueEstimate
from modelbench.suts import ModelGaugeSut
from test_static_site_generator import benchmark_score


def encode(o):
    return json.dumps(o, cls=BenchmarkScoreEncoder, indent=4)


def encode_and_parse(o):
    s = encode(o)
    return json.loads(s)


def test_sut():
    sut = ModelGaugeSut.for_key("mistral-7b")
    assert encode_and_parse(sut) == {"uid": "mistral-7b"}
    sut.instance(MagicMock())
    with_initialization = encode_and_parse(sut)
    assert "uid" in with_initialization
    assert "initialization" in with_initialization
    assert encode_and_parse(sut) == with_initialization


def test_anonymous_sut():
    j = encode_and_parse(FakeSut("a_sut-v1.0"))
    assert j["uid"] == "a_sut-v1.0"


def test_value_estimate():
    ve = ValueEstimate.make(0.5, 1000)
    j = encode_and_parse(ve)
    assert j["lower"] == ve.lower
    assert j["estimate"] == ve.estimate
    assert j["upper"] == ve.upper
    assert j["samples"] == ve.samples


def test_hazard_definition_without_tests_loaded():
    hazard = SafeCaeHazard()
    j = encode_and_parse(hazard)
    assert j["uid"] == hazard.uid
    assert "tests" not in j
    assert j["reference_standard"] == hazard.reference_standard()


def test_hazard_definition_with_tests_loaded():
    hazard = SafeCaeHazard()
    hazard.tests({"together": {"api_key": "ignored"}})
    j = encode_and_parse(hazard)
    assert j["uid"] == hazard.uid
    assert j["tests"] == ["safe-cae"]
    assert j["reference_standard"] == hazard.reference_standard()


def test_v1_hazard_definition_with_tests_loaded():
    hazard = SafeDfmHazardV1()
    hazard.tests({"together": {"api_key": "ignored"}})
    j = encode_and_parse(hazard)
    assert j["uid"] == hazard.uid
    assert j["tests"] == ["safe-dfm-1.0"]
    assert j["reference_standard"] == hazard.reference_standard()


def test_benchmark_definition():
    j = encode_and_parse(GeneralPurposeAiChatBenchmark())
    assert j["uid"] == "general_purpose_ai_chat_benchmark-0.5"
    assert "safe_cae_hazard-0.5" in [i["uid"] for i in j["hazards"]]


def test_hazard_score():
    ve = ValueEstimate.make(1.0, 100000)
    hs = HazardScore(hazard_definition=SafeCaeHazard(), score=ve, test_scores={"cae": ve}, exceptions=0)
    j = encode_and_parse(hs)
    assert j["hazard_definition"]["uid"] == SafeCaeHazard().uid
    assert j["score"]["samples"] == ve.samples
    assert j["test_scores"]["cae"]["samples"] == ve.samples
    assert j["numeric_grade"] == 5
    assert j["text_grade"] == "L"
    assert j["exceptions"] == 0


def test_benchmark_score(benchmark_score):
    j = encode_and_parse(benchmark_score)
    assert "benchmark_definition" not in j  # it's already higher up in the tree; no need to duplicate
    assert j["sut"]["uid"] == benchmark_score.sut.key
    assert len(j["hazard_scores"]) == len(benchmark_score.hazard_scores)
    assert j["end_time"] == str(benchmark_score.end_time)
    assert j["numeric_grade"] == benchmark_score.numeric_grade()
    assert j["text_grade"] == benchmark_score.text_grade()


def test_benchmark_score_initialization_record(benchmark_score):
    mock_method = Mock()
    mock_method.return_value = InitializationRecord(
        module="a_module", class_name="a_class", args=["arg1", "arg2"], kwargs={"kwarg1": "a_value"}
    )
    benchmark_score.sut.instance_initialization = mock_method
    j = encode_and_parse(benchmark_score)
    print(j)
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
