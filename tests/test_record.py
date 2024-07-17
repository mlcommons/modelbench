import json
import platform
import re
from unittest.mock import Mock

from modelgauge.load_plugins import load_plugins
from modelgauge.record_init import InitializationRecord

import modelbench
from modelbench.benchmarks import GeneralPurposeAiChatBenchmark
from modelbench.hazards import HazardScore, SafeCaeHazard
from modelbench.modelgauge_runner import ModelGaugeSut
from modelbench.record import BenchmarkScoreEncoder, benchmark_run_record
from modelbench.run import score_benchmarks
from modelbench.scoring import ValueEstimate
from test_static_site_generator import benchmark_score


def encode(o):
    return json.dumps(o, cls=BenchmarkScoreEncoder, indent=4)


def encode_and_parse(o):
    s = encode(o)
    return json.loads(s)


def test_sut():
    assert encode_and_parse(ModelGaugeSut.ALPACA_7B) == {"uid": "alpaca-7b"}


def test_value_estimate():
    ve = ValueEstimate.make(0.5, 1000)
    j = encode_and_parse(ve)
    assert j["lower"] == ve.lower
    assert j["estimate"] == ve.estimate
    assert j["upper"] == ve.upper
    assert j["samples"] == ve.samples


def test_hazard_definition():
    assert encode_and_parse(SafeCaeHazard()) == "safe_cae_hazard-0.5"


def test_benchmark_definition():
    j = encode_and_parse(GeneralPurposeAiChatBenchmark())
    assert j["uid"] == "general_purpose_ai_chat_benchmark-0.5"
    assert "safe_cae_hazard-0.5" in j["hazards"]


def test_hazard_score():
    ve = ValueEstimate.make(1.0, 100000)
    hs = HazardScore(hazard_definition=SafeCaeHazard(), score=ve, test_scores={"cae": ve})
    j = encode_and_parse(hs)
    assert j["hazard_definition"] == SafeCaeHazard().uid
    assert j["score"]["samples"] == ve.samples
    assert j["test_scores"]["cae"]["samples"] == ve.samples
    assert j["numeric_grade"] == 5
    assert j["text_grade"] == "L"


def test_benchmark_score(benchmark_score):
    j = encode_and_parse(benchmark_score)
    assert j["benchmark_definition"]["uid"] == benchmark_score.benchmark_definition.uid
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
    code = r["_metadata"]["code"]
    assert code["git_version"].startswith("git version 2")
    assert code["origin"] == "git@github.com:mlcommons/modelbench.git"
    assert re.match(r"v[.0-9]+-\d+-[a-z0-9]{8}", code["code_version"])
    assert isinstance(code["changed_files"], list)  # hard to be more specific here


def test_benchmark_code_record_without_git(benchmark_score):
    def fnf(*args):
        raise FileNotFoundError()

    modelbench.record.run_command = fnf
    r = benchmark_run_record(benchmark_score)
    code = r["_metadata"]["code"]
    assert code["error"] == "git command not found"
