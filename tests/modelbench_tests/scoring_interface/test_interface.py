import json
from importlib.resources import files

import pytest
from pydantic import ValidationError

from modelbench.scoring_interface.interface import (
    CONTRACT_MODELS,
    PublicationBlocked,
    SafetyScoringResult,
    SecurityScoringResult,
)
from scoring_science.safety_v_1_2_1_scoring import SAFETY_1_2_1_SCORING_FILES
from scoring_science.security_v_1_1_scoring import SECURITY_1_1_SCORING_FILES

ACCEPTED_FIT_DIGESTS = {
    ("safety", "official"): "81842dd6d8b9785562c7779d557b6ae5cdc4d7d3fe53958adcd157e543b23db3",
    ("safety", "practice"): "240ed1d092512c64b1265266559c484497f93061cb743ca49e453ccb516f35a2",
    ("security", "official"): "b423d9fc08f8ddd061d06df19feee4fb96d5390765a6794d8dc2a46a9e5ea946",
    ("security", "practice"): "f02ce7f23182348ce42e6b10f7a691c3e2a06cc3f8923346be4fa02ebbad0ed0",
}

SCHEMA_ANNEX = files("scoring_science").joinpath("contract_schemas.json")


def safety_result_dict():
    block = {
        "reference_percentile_score": 90.5,
        "numeric_grade": None,
        "mean_violation_probability": 0.2,
        "errors": [],
    }
    notice = {
        "code": "estimate_at_limit",
        "message": "held at the supported limit",
        "nonfatal": True,
        "hazard": "vcr",
        "limit": 4.0,
    }
    return {
        "sut_uid": "fabricated-sut",
        "complete_responses": 240,
        "score": block,
        "hazard_scores": [
            {"hazard": "vcr", "num_scored_items": 20, "score": {**block, "errors": [notice]}},
            {"hazard": "cse", "num_scored_items": 220, "score": dict(block)},
        ],
        "difficulty_and_discrimination_sha256": "f" * 64,
    }


def security_result_dict():
    def block(probability):
        return {
            "reference_percentile_score": 90.0 - probability,
            "numeric_grade": None,
            "mean_violation_probability": probability,
            "errors": [],
        }

    return {
        "sut_uid": "fabricated-sut",
        "complete_pairs": 200,
        "scores": {"naive": block(0.2), "jailbreak": block(0.8)},
        "hazard_scores": [
            {"hazard": "vcr", "num_scored_items": 20, "naive": block(0.2), "jailbreak": block(0.8)},
        ],
        "difficulty_and_discrimination_sha256": "f" * 64,
    }


def test_valid_safety_result_parses():
    result = SafetyScoringResult.model_validate(safety_result_dict())
    assert result.hazard_scores[0].score.errors[0].hazard == "vcr"
    assert result.hazard_scores[0].score.errors[0].nonfatal is True
    assert result.score.numeric_grade is None


def test_valid_security_result_parses():
    result = SecurityScoringResult.model_validate(security_result_dict())
    assert result.scores.jailbreak.reference_percentile_score < result.scores.naive.reference_percentile_score
    assert result.hazard_scores[0].naive.mean_violation_probability == pytest.approx(0.2)


def test_results_are_immutable():
    result = SafetyScoringResult.model_validate(safety_result_dict())
    with pytest.raises(ValidationError):
        result.complete_responses = 0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda raw: raw.pop("complete_responses"),
        lambda raw: raw.update(calibration_theta=1.0),
        lambda raw: raw["score"].update(theta=0.1),
        lambda raw: raw["score"].update(reference_percentile_score="90.5"),
        lambda raw: raw["score"].update(numeric_grade=3),
        lambda raw: raw["hazard_scores"][0].update(weight=2),
        lambda raw: raw["hazard_scores"][0]["score"]["errors"][0].update(nonfatal=1),
        lambda raw: raw["hazard_scores"][0]["score"]["errors"][0].pop("limit"),
        lambda raw: raw.pop("difficulty_and_discrimination_sha256"),
    ],
    ids=[
        "missing-field",
        "extra-top-level-field",
        "extra-block-field",
        "string-for-score",
        "grade-returns",
        "extra-hazard-field",
        "int-for-bool",
        "missing-notice-field",
        "missing-fit-digest",
    ],
)
def test_drifted_safety_result_fails_at_parse(mutation):
    raw = safety_result_dict()
    mutation(raw)
    with pytest.raises(ValidationError):
        SafetyScoringResult.model_validate(raw)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda raw: raw["scores"].pop("jailbreak"),
        lambda raw: raw["scores"].update(calibration={}),
        lambda raw: raw["hazard_scores"][0].pop("naive"),
        lambda raw: raw.update(complete_pairs="200"),
        lambda raw: raw.pop("difficulty_and_discrimination_sha256"),
    ],
    ids=["missing-condition", "extra-condition", "missing-hazard-condition", "string-for-count", "missing-fit-digest"],
)
def test_drifted_security_result_fails_at_parse(mutation):
    raw = security_result_dict()
    mutation(raw)
    with pytest.raises(ValidationError):
        SecurityScoringResult.model_validate(raw)


@pytest.mark.parametrize("prompt_set", ["official", "practice"])
def test_safety_fit_is_the_accepted_fit(prompt_set):
    _, digest = SAFETY_1_2_1_SCORING_FILES[prompt_set]["en_us"].load()
    assert digest == ACCEPTED_FIT_DIGESTS[("safety", prompt_set)]


@pytest.mark.parametrize("prompt_set", ["official", "practice"])
def test_security_fit_is_the_accepted_fit(prompt_set):
    _, digest = SECURITY_1_1_SCORING_FILES[prompt_set]["en_us"].load()
    assert digest == ACCEPTED_FIT_DIGESTS[("security", prompt_set)]


def test_schema_annex_is_the_one_the_models_generate():
    schemas = {model.__name__: model.model_json_schema() for model in CONTRACT_MODELS}
    assert json.loads(SCHEMA_ANNEX.read_text()) == schemas, (
        "The contract models changed; regenerate the science team's annex in the "
        "candela-science repo with `uv run modelbench scoring-schema -o src/scoring_science/contract_schemas.json`"
    )


def test_publication_blocked_names_hazards_and_counts():
    blocked = PublicationBlocked("fabricated-sut", ["cse"], {"cse": {"expected": 20, "incomplete": 1}})
    assert blocked.hazards == ["cse"]
    assert "cse (1 of 20)" in str(blocked)
    paired = PublicationBlocked("s", ["vcr"], {"vcr": {"expected_pairs": 20, "incomplete_pairs": 3}})
    assert "vcr (3 of 20)" in str(paired)
