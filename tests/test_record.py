import json
import platform
import re

from modelgauge.load_plugins import load_plugins

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
    print(s)
    return json.loads(s)

def test_sut():
    assert encode_and_parse(ModelGaugeSut.ALPACA_7B) == "alpaca-7b"


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
    assert j["sut"] == benchmark_score.sut.key
    assert len(j["hazard_scores"]) == len(benchmark_score.hazard_scores)
    assert j["end_time"] == str(benchmark_score.end_time)
    assert j["numeric_grade"] == benchmark_score.numeric_grade()
    assert j["text_grade"] == benchmark_score.text_grade()


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


# TODO add initialization records


# TODO Figure out how to exclude this from normal runs (or to include cache)
def test_save_a_full_run():
    load_plugins()
    benchmark = GeneralPurposeAiChatBenchmark()
    benchmark_scores = score_benchmarks(
        [benchmark], [ModelGaugeSut.ALPACA_7B, ModelGaugeSut.LLAMA_2_7B], 30, False, True
    )
    score = benchmark_scores[0]

    with open(f"benchmark_record-{benchmark.uid}.json", "w") as f:
        output = benchmark_run_record(score)
        json.dump(output, f, cls=BenchmarkScoreEncoder, indent=4)
