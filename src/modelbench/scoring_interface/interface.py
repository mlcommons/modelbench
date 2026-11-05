from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from modelgauge.annotation import ViolationProbabilityAnnotation


class _ContractModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _ResultModel(_ContractModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


SecurityCondition = Literal["naive", "jailbreak"]


# Requests


class SafetyFormRow(_ContractModel):
    prompt_uid: str
    hazard: str


class SecurityFormRow(_ContractModel):
    prompt_uid: str
    naive_prompt_uid: str
    prompt_condition: SecurityCondition
    hazard: str


class EvaluationRecord(_ContractModel):
    sut_uid: str
    prompt_uid: str
    is_valid: bool | None = None
    violation_probability: float | None = None


def evaluation_row(item, sut_uid: str, benchmark: str) -> EvaluationRecord:
    if item.sut.uid != sut_uid:
        raise ValueError(f"Finished item for prompt {item.source_id()} belongs to SUT {item.sut.uid}, not {sut_uid}")
    if len(item.annotations) != 1:
        raise ValueError(f"Prompt {item.source_id()} has {len(item.annotations)} evaluations; {benchmark} expects one")
    annotation = next(iter(item.annotations.values()))
    if not isinstance(annotation, ViolationProbabilityAnnotation):
        raise ValueError(
            f"Prompt {item.source_id()} has a yes/no evaluation; {benchmark} scoring needs the probability "
            "evaluator (run with --evaluator private)"
        )
    return EvaluationRecord(
        sut_uid=sut_uid,
        prompt_uid=item.source_id(),
        is_valid=annotation.is_valid,
        violation_probability=annotation.violation_probability,
    )


# Results


class ScoreNotice(_ResultModel):
    code: str
    message: str
    nonfatal: bool
    hazard: str
    limit: float


class ScoreBlock(_ResultModel):
    reference_percentile_score: float
    numeric_grade: None = None
    mean_violation_probability: float
    errors: list[ScoreNotice]


class HazardScoreEntry(_ResultModel):
    hazard: str
    num_scored_items: int
    score: ScoreBlock


class SafetyScoringResult(_ResultModel):
    sut_uid: str
    complete_responses: int
    score: ScoreBlock
    hazard_scores: list[HazardScoreEntry]
    difficulty_and_discrimination_sha256: str


class PairedScoreBlocks(_ResultModel):
    naive: ScoreBlock
    jailbreak: ScoreBlock


class SecurityHazardScoreEntry(_ResultModel):
    hazard: str
    num_scored_items: int
    naive: ScoreBlock
    jailbreak: ScoreBlock


class SecurityScoringResult(_ResultModel):
    sut_uid: str
    complete_pairs: int
    scores: PairedScoreBlocks
    hazard_scores: list[SecurityHazardScoreEntry]
    difficulty_and_discrimination_sha256: str


class PublicationBlocked(ValueError):
    def __init__(self, sut_uid: str, hazards: Sequence[str], coverage: Mapping[str, Mapping[str, Any]]):
        self.sut_uid = sut_uid
        self.hazards = list(hazards)
        self.coverage = coverage
        detail = ", ".join(f"{hazard} ({self._counts(coverage[hazard])})" for hazard in self.hazards)
        super().__init__(f"Publication blocked for {sut_uid}: responses are incomplete in {detail}")

    @staticmethod
    def _counts(counts: Mapping[str, Any]) -> str:
        incomplete = counts.get("incomplete", counts.get("incomplete_pairs"))
        expected = counts.get("expected", counts.get("expected_pairs"))
        return f"{incomplete} of {expected}"
