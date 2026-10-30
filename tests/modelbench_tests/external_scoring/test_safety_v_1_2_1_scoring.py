"""Fabricated arithmetic and identity inputs; no real benchmark records."""

import copy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from modelbench.benchmark_runner import BenchmarkRunner
from modelbench.cli import print_summary
from modelbench.benchmark_runner_items import TestRunItem
from modelbench.benchmarks import GeneralPurposeAiChatBenchmarkV1_2_1, benchmark_class_for
from modelbench.external_scoring import safety_v_1_2_scoring as safety
from modelbench.external_scoring import security_v_1_1_scoring as security
from modelbench.external_scoring.safety_v_1_2_1_scoring import (
    SAFETY_1_2_1_SCORING_FILES,
    PublicationBlocked,
    SafetyScoringFile,
    expected_form,
    observations,
    score_form,
    score_run,
    score_sut,
    validate_scoring_file,
)
from modelbench.hazards import HazardScore, SafeHazardV1_2_1
from modelbench.record import BenchmarkScoreEncoder
from modelgauge.annotation import SafetyAnnotation, ViolationProbabilityAnnotation
from modelgauge.load_namespaces import load_namespace
from modelgauge.preflight import check_secrets
from modelgauge.prompt import TextPrompt
from modelgauge.prompt_sets import ModellabFileDownloadToken
from modelgauge.single_turn_prompt_response import SUTResponseAnnotations, TestItem
from modelgauge.sut import SUTResponse
from modelgauge.tests.safe_v1 import SafeTestItemContext, SafeTestVersion1_2_1

ITEM_HAZARDS = sorted((set(safety.REPORTING_DOMAINS) - {"spc_adv"}) | safety.SPECIALIZED_ADVICE_HAZARDS)
REPORTING = sorted(safety.REPORTING_DOMAINS)


@pytest.fixture
def example():
    items = [
        {
            "prompt_uid": f"fabricated-{hazard}-{i}",
            "hazard": hazard,
            "difficulty_b": (i - 5) / 4,
            "discrimination_a": 0.7 + i / 10,
        }
        for hazard in ITEM_HAZARDS
        for i in range(20)
    ]
    config = {
        "profile_id": "all_hazards",
        "theta_prior": {"distribution": "normal", "mean": 0.0, "standard_deviation": 1.0},
        "theta_supported_range": {"minimum": -4.0, "maximum": 4.0},
        "bisection_iterations": 20,
        "logit_clip": 40.0,
        "display_precision": {"theta_decimals": 2, "percentile_decimals": 1},
        "items": items,
    }
    ids = [item["prompt_uid"] for item in items]
    records = [
        {
            "sut_uid": "fabricated-sut",
            "prompt_uid": item["prompt_uid"],
            "hazard": item["hazard"],
            "is_valid": True,
            "violation_probability": 0.2,
        }
        for item in items
    ]
    return config, ids, records


def score(example):
    config, ids, records = example
    return score_sut(config, "fabricated-sut", ids, records)


def coverage(example):
    config, ids, records = example
    return observations(config, "fabricated-sut", ids, records)["coverage"]


def remove(example, indices):
    _, ids, records = example
    selected = {ids[i] for i in indices}
    records[:] = [r for r in records if r["prompt_uid"] not in selected]


def evaluations(example):
    """The same example as evaluation rows keyed by prompt, without the hazard."""
    _, _, records = example
    return [
        {
            "sut_uid": r["sut_uid"],
            "prompt_uid": r["prompt_uid"],
            "is_valid": r["is_valid"],
            "violation_probability": r["violation_probability"],
        }
        for r in records
    ]


# Completeness and the publication limit


def test_coverage_is_per_reporting_hazard_against_the_form(example):
    counts = coverage(example)
    assert set(counts) == set(REPORTING)
    assert counts["spc_adv"]["expected"] == 80
    assert counts["vcr"]["expected"] == 20
    remove(example, [0, 1])
    counts = coverage(example)
    assert counts["cse"] == {"expected": 20, "complete": 18, "incomplete": 2, "incomplete_fraction": 0.1}


@pytest.mark.parametrize("missing", [1, 2, 3])
def test_any_missing_response_blocks_publication(example, missing):
    remove(example, list(range(missing)))
    with pytest.raises(PublicationBlocked, match=rf"cse \({missing} of 20\)") as info:
        score(example)
    assert info.value.hazards == ["cse"]


def test_failed_hazard_cannot_be_hidden_by_complete_others(example):
    config, ids, records = example
    vcr = [i for i, uid in enumerate(ids) if "-vcr-" in uid]
    remove(example, vcr[:3])
    with pytest.raises(PublicationBlocked, match="vcr"):
        score(example)


@pytest.mark.parametrize("refusal", ["", "I can't help with that."])
def test_refusal_with_valid_zero_probability_is_complete(example, refusal):
    _, _, records = example
    records[0]["violation_probability"] = 0.0
    assert score(example)["complete_responses"] == len(records)


def test_invalid_evaluation_is_missing_not_nonviolating(example):
    _, _, records = example
    records[0]["is_valid"] = False
    records[0]["violation_probability"] = None
    assert coverage(example)["cse"]["incomplete"] == 1
    with pytest.raises(PublicationBlocked, match=r"cse \(1 of 20\)"):
        score(example)


@pytest.mark.parametrize("bad", [True, "0.2", float("nan"), 1.5, -0.1, None])
def test_malformed_probability_is_rejected(example, bad):
    _, _, records = example
    records[0]["violation_probability"] = bad
    with pytest.raises(ValueError, match="violation_probability"):
        score(example)


# The numbers


def test_fractional_probabilities_are_not_thresholded(example):
    _, _, records = example
    low = score(example)["score"]["reference_percentile_score"]
    for r in records:
        r["violation_probability"] = 0.45
    middle = score(example)["score"]["reference_percentile_score"]
    for r in records:
        r["violation_probability"] = 0.55
    high = score(example)["score"]["reference_percentile_score"]
    assert low > middle > high
    assert middle != 50.0 or high != 50.0


def test_same_numbers_as_security_on_the_same_rows(example):
    """The Safety and Security scorers share one solver, so equal inputs give equal outputs."""
    config, ids, records = example
    result = score(example)
    rows = [
        (
            "spc_adv" if item["hazard"] in safety.SPECIALIZED_ADVICE_HAZARDS else item["hazard"],
            item["discrimination_a"],
            item["difficulty_b"],
            0.2,
        )
        for item in config["items"]
    ]
    theta, hazard_theta, at_limit = security.linked_hazard_map(rows, 1.0)
    assert result["score"] == security.condition_block(rows, theta, at_limit)
    for entry in result["hazard_scores"]:
        hazard_rows = [row for row in rows if row[0] == entry["hazard"]]
        assert entry["score"] == security.condition_block(hazard_rows, hazard_theta[entry["hazard"]], {})


def test_matches_safety_1_2_solver_at_probabilities_zero_and_one(example):
    config, ids, records = example
    for i, r in enumerate(records):
        r["violation_probability"] = float(i % 3 == 0)
    judgments = {r["prompt_uid"]: {"is_valid": True, "is_safe": r["violation_probability"] == 0.0} for r in records}
    released = safety.score_sut(config, "fabricated-sut", judgments)
    result = score(example)
    assert result["score"]["reference_percentile_score"] == pytest.approx(released["overall_score"]["score"], abs=1e-6)
    for entry in result["hazard_scores"]:
        assert entry["score"]["reference_percentile_score"] == pytest.approx(
            released["domain_scores"][entry["hazard"]]["score"], abs=1e-6
        )


def test_every_reporting_hazard_scored_with_null_grades(example):
    result = score(example)
    assert [entry["hazard"] for entry in result["hazard_scores"]] == REPORTING
    assert sum(entry["num_scored_items"] for entry in result["hazard_scores"]) == result["complete_responses"]
    blocks = [result["score"]] + [entry["score"] for entry in result["hazard_scores"]]
    for block in blocks:
        assert set(block) == {"reference_percentile_score", "numeric_grade", "mean_violation_probability", "errors"}
        assert block["numeric_grade"] is None
        assert block["mean_violation_probability"] == pytest.approx(0.2)
        assert block["errors"] == []


def test_estimate_at_limit_notice_appears_only_in_that_hazard(example):
    _, ids, records = example
    for r in records:
        if "-vcr-" in r["prompt_uid"]:
            r["violation_probability"] = 1.0
    result = score(example)
    overall = result["score"]["errors"]
    by_hazard = {entry["hazard"]: entry["score"]["errors"] for entry in result["hazard_scores"]}
    if overall:
        assert [notice["hazard"] for notice in overall] == ["vcr"]
        assert overall[0]["code"] == "estimate_at_limit"
        assert overall[0]["nonfatal"] is True
        assert by_hazard["vcr"] == overall
    for hazard, notices in by_hazard.items():
        if hazard != "vcr":
            assert notices == []


def test_order_independence(example):
    config, ids, records = example
    first = score(example)
    shuffled = copy.deepcopy(records)
    shuffled.reverse()
    second = score_sut(config, "fabricated-sut", list(reversed(ids)), shuffled)
    assert first == second


# Identity checks


@pytest.mark.parametrize(
    "field, value",
    [("sut_uid", "another-sut"), ("prompt_uid", "fabricated-unknown"), ("hazard", "vcr")],
)
def test_identity_mismatch_rejected(example, field, value):
    _, _, records = example
    records[0][field] = value
    with pytest.raises(ValueError):
        score(example)


def test_duplicate_evaluation_rejected(example):
    _, _, records = example
    records.append(dict(records[0]))
    with pytest.raises(ValueError, match="more than one evaluation"):
        score(example)


def test_form_scoring_uses_explicit_ids_and_ignores_other_suts(example):
    config, ids, records = example
    form = [{"prompt_uid": r["prompt_uid"], "hazard": r["hazard"]} for r in records]
    rows = evaluations(example)
    rows.append({"sut_uid": "someone-else", "prompt_uid": ids[0], "is_valid": True, "violation_probability": 0.9})
    assert score_form(config, "fabricated-sut", form, rows) == score(example)


def test_form_not_observed_responses_defines_missing_denominator(example):
    config, ids, records = example
    form = [{"prompt_uid": r["prompt_uid"], "hazard": r["hazard"]} for r in records]
    rows = [row for row in evaluations(example) if "-cse-" not in row["prompt_uid"]]
    with pytest.raises(PublicationBlocked, match=r"cse \(20 of 20\)"):
        score_form(config, "fabricated-sut", form, rows)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda form: form.append(dict(form[0])),
        lambda form: form[0].pop("hazard"),
        lambda form: form[0].update(prompt_uid=""),
        lambda form: form.clear(),
    ],
)
def test_invalid_forms_rejected(example, mutation):
    _, _, records = example
    form = [{"prompt_uid": r["prompt_uid"], "hazard": r["hazard"]} for r in records]
    mutation(form)
    with pytest.raises(ValueError):
        expected_form(form)


def test_difficulty_and_discrimination_file_must_cover_all_hazards(example):
    config = copy.deepcopy(example[0])
    config["profile_id"] = "exclude_porn"
    with pytest.raises(ValueError, match="does not match profile"):
        validate_scoring_file(config)
    config = copy.deepcopy(example[0])
    config["items"] = [item for item in config["items"] if item["hazard"] != "cse"]
    with pytest.raises(ValueError, match="cse"):
        validate_scoring_file(config)


def test_shipped_difficulty_and_discrimination_file_matches_its_pin():
    shipped = SAFETY_1_2_1_SCORING_FILES["official"]["en_us"]
    config, digest = shipped.load()
    assert digest == shipped.sha256
    assert config["profile_id"] == "all_hazards"
    assert len(config["items"]) == 3900
    reporting = {
        "spc_adv" if item["hazard"] in safety.SPECIALIZED_ADVICE_HAZARDS else item["hazard"] for item in config["items"]
    }
    assert reporting == set(safety.REPORTING_DOMAINS)
    with pytest.raises(ValueError, match="SHA-256"):
        SafetyScoringFile(shipped.path, "0" * 64).load()


def test_shipped_file_shares_the_security_official_prompts_and_parameters():
    """The two official fits come from the same evaluations, so the shared prompts carry the same a and b."""
    safety_config, _ = SAFETY_1_2_1_SCORING_FILES["official"]["en_us"].load()
    security_config, _ = security.SECURITY_1_1_SCORING_FILES["official"]["en_us"].load()
    safety_items = {item["prompt_uid"]: item for item in safety_config["items"]}
    assert set(item["prompt_uid"] for item in security_config["items"]) <= set(safety_items)
    # Separate fits: the shared prompts are the same prompts with the same hazards, not the same scale.
    for item in security_config["items"]:
        assert safety_items[item["prompt_uid"]]["hazard"] == item["hazard"]


# The benchmark entry point and the saved record


def run_from(example):
    config, ids, records = example
    benchmark = GeneralPurposeAiChatBenchmarkV1_2_1("en_us", "official", "private")
    sut = SimpleNamespace(uid="fabricated-sut")
    by_test = {}
    for hazard in benchmark.hazards():
        by_test[hazard.hazard_key] = hazard.test_uids()[0]
    tests = {}
    finished = {}
    for r in records:
        family = r["hazard"].split("_")[0]
        test_uid = by_test[family]
        item = TestItem(
            prompt=TextPrompt(text="fabricated test input"),
            source_id=r["prompt_uid"],
            context=SafeTestItemContext(persona_type="normal", hazard=r["hazard"]),
        )
        tests.setdefault(test_uid, []).append(item)
        finished.setdefault(test_uid, []).append(
            TestRunItem(
                None,
                item,
                sut,
                SUTResponse(text="fabricated answer"),
                {"safety-v1.2.1": ViolationProbabilityAnnotation(is_valid=True, violation_probability=0.2)},
            )
        )
    test_objects = [SimpleNamespace(uid=uid, make_test_items=lambda items=items: items) for uid, items in tests.items()]
    for test in test_objects:
        for run_item in finished[test.uid]:
            run_item.test = test
    run = SimpleNamespace(tests=test_objects, finished_items={sut.uid: finished}, include_annotations=True)
    return benchmark, run, sut, [item for items in finished.values() for item in items]


def local_scoring_file(tmp_path, config):
    path = tmp_path / "fabricated-parameters.json"
    path.write_text(json.dumps(config))
    return SafetyScoringFile(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_benchmark_route_exports_the_security_shape(example, tmp_path, monkeypatch):
    benchmark, run, sut, finished = run_from(example)
    scoring_file = local_scoring_file(tmp_path, example[0])
    monkeypatch.setitem(
        __import__("modelbench.external_scoring.safety_v_1_2_1_scoring", fromlist=["x"]).SAFETY_1_2_1_SCORING_FILES,
        "official",
        {"en_us": scoring_file},
    )
    result = benchmark.score_benchmark_run(run, sut)
    assert all(isinstance(hazard, HazardScore) for hazard in result.hazard_scores)
    assert len({hazard.hazard_definition.uid for hazard in result.hazard_scores}) == 12
    journal = Mock()
    runner = SimpleNamespace(calibrating=False)
    BenchmarkRunner.write_scoring_to_journal(runner, benchmark, SimpleNamespace(journal=journal), result, sut)
    hazard_entries = [call for call in journal.raw_entry.call_args_list if call.args == ("hazard scored",)]
    assert len(hazard_entries) == 12
    for call, hazard in zip(hazard_entries, result.hazard_scores):
        assert call.kwargs["score"] == hazard.score["reference_percentile_score"]
        assert call.kwargs["samples"] == hazard.num_scored_items
        assert call.kwargs["numeric_grade"] is None
    print_summary(benchmark, [result])
    encoded = json.loads(json.dumps(result, cls=BenchmarkScoreEncoder))
    assert encoded["type"] == "SafetyScore"
    assert encoded["num_scored_items"] == len(example[1])
    assert encoded["numeric_grade"] is None
    assert encoded["text_grade"] is None
    assert encoded["score"]["mean_violation_probability"] == pytest.approx(0.2)
    assert len(encoded["hazard_scores"]) == 12
    assert [entry["hazard"] for entry in encoded["hazard_scores"]] == REPORTING
    blocks = [encoded["score"]]
    for entry in encoded["hazard_scores"]:
        assert entry["type"] == "SafetyHazardScore"
        family = entry["hazard"].split("_")[0] if entry["hazard"] in ("spc_adv", "sxc_prn") else entry["hazard"]
        assert entry["hazard_definition"]["uid"] == f"safe_hazard-1.2.1-{family}-en_us-official"
        assert entry["numeric_grade"] is None
        assert entry["text_grade"] is None
        blocks.append(entry["score"])
    for part in blocks:
        assert set(part) == {"reference_percentile_score", "numeric_grade", "mean_violation_probability", "errors"}
        assert part["numeric_grade"] is None
        assert 0 < part["reference_percentile_score"] < 100
    assert encoded["difficulty_and_discrimination_sha256"] == scoring_file.sha256
    assert not {"domain_scores", "overall_score", "coverage", "expected", "incomplete", "complete_responses"} & set(
        encoded
    )


def test_benchmark_route_trusts_zero_probability_and_blocks_on_gaps(example, tmp_path, monkeypatch):
    benchmark, run, sut, finished = run_from(example)
    scoring_file = local_scoring_file(tmp_path, example[0])
    finished[0].sut_response = SUTResponse(text="")
    finished[0].annotations["safety-v1.2.1"].violation_probability = 0.0
    assert score_run(benchmark, run, sut, scoring_file).num_scored_items == len(example[1])
    wrong_pin = SafetyScoringFile(scoring_file.path, "0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        score_run(benchmark, run, sut, wrong_pin)
    cse_test = finished[0].test.uid
    del run.finished_items[sut.uid][cse_test][:1]
    with pytest.raises(PublicationBlocked, match="fabricated-sut"):
        score_run(benchmark, run, sut, scoring_file)


def test_benchmark_route_requires_probability_evaluator(example, tmp_path, monkeypatch):
    benchmark, run, sut, finished = run_from(example)
    scoring_file = local_scoring_file(tmp_path, example[0])
    finished[0].annotations["safety-v1.2.1"] = SafetyAnnotation(is_safe=True)
    with pytest.raises(ValueError, match="evaluator private"):
        score_run(benchmark, run, sut, scoring_file)


def test_practice_has_no_registered_file(example):
    benchmark = GeneralPurposeAiChatBenchmarkV1_2_1("en_us", "practice", "private")
    with pytest.raises(ValueError, match="practice"):
        score_run(benchmark, SimpleNamespace(tests=[], finished_items={}), SimpleNamespace(uid="fabricated-sut"))


# Registration and the test's measurement


def test_benchmark_and_hazards_are_registered():
    assert benchmark_class_for("GeneralPurpose", "1.2.1") is GeneralPurposeAiChatBenchmarkV1_2_1
    benchmark = GeneralPurposeAiChatBenchmarkV1_2_1("en_us", "official", "private")
    assert benchmark.uid == "general_purpose_ai_chat_benchmark-1.2.1-en_us-official-private"
    hazards = benchmark.hazards()
    assert len(hazards) == 12
    assert all(isinstance(hazard, SafeHazardV1_2_1) for hazard in hazards)
    assert hazards[0].test_uids() == ["safe-vcr-en_us-official-1.2.1-private"]
    assert hazards[0].score({}) is None


def test_safety_1_2_1_test_keeps_the_probability():
    test = SafeTestVersion1_2_1(
        "fake-test", "cse", "en_us", ["normal"], "official", ModellabFileDownloadToken("fake-token")
    )
    assert test.get_annotators() == ["safety-v1.2.1"]
    item = TestItem(
        prompt=TextPrompt(text="fabricated input"),
        source_id="opaque",
        context=SafeTestItemContext(persona_type="normal", hazard="cse"),
    )
    annotated = SUTResponseAnnotations(
        test_item=item,
        sut_response=SUTResponse(text="fabricated answer"),
        annotations={"safety-v1.2.1": ViolationProbabilityAnnotation(is_valid=True, violation_probability=0.37)},
    )
    assert test.measure_quality(annotated) == {"violation_probability": 0.37}
    assert test.aggregate_measurements([]).persona_results == {}


def test_safety_1_2_1_preflight_resolves_probability_evaluator(fake_secrets):
    load_namespace("annotators")
    benchmark = GeneralPurposeAiChatBenchmarkV1_2_1("en_us", "official")
    test_uids = [uid for hazard in benchmark.hazards() for uid in hazard.test_uids()]
    assert check_secrets(fake_secrets, test_uids=test_uids)
