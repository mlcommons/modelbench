from datetime import datetime
from typing import Mapping

from scoring_science import safety_v_1_2_1_scoring as science
from scoring_science.safety_v_1_2_1_scoring import PublicationBlocked as SciencePublicationBlocked

from modelbench.benchmark_score import BaseBenchmarkScore
from modelbench.hazards import HazardScore
from modelbench.scoring import NoGradesMixin
from modelbench.scoring_interface.interface import (
    SafetyFormRow,
    PublicationBlocked,
    ScoreBlock,
    SafetyScoringResult,
    evaluation_row,
)


def hazard_family(hazard: str) -> str:
    return hazard.split("_")[0] if hazard in ("spc_adv", "sxc_prn") else hazard


def form_row(item) -> SafetyFormRow:
    context = item.context
    hazard = context.get("hazard") if isinstance(context, Mapping) else getattr(context, "hazard", None)
    if hazard is None:
        raise ValueError(f"Prompt {item.source_id} has no hazard in its context")
    return SafetyFormRow(prompt_uid=item.source_id, hazard=hazard)


class SafetyHazardScore(NoGradesMixin, HazardScore):
    hazard: str
    score: ScoreBlock

    def actual_score(self) -> float:
        return self.score.reference_percentile_score


class SafetyScore(NoGradesMixin, BaseBenchmarkScore):
    def __init__(self, benchmark, sut, result: SafetyScoringResult, end_time):
        definitions = {hazard.hazard_key: hazard for hazard in benchmark.hazards()}
        hazard_scores = [
            SafetyHazardScore(
                hazard_definition=definitions[hazard_family(entry.hazard)],
                hazard=entry.hazard,
                score=entry.score,
                test_scores={},
                exceptions=0,
                num_scored_items=entry.num_scored_items,
            )
            for entry in result.hazard_scores
        ]
        super().__init__(benchmark, sut, hazard_scores, end_time)
        self.score = result.score
        self.num_scored_items = result.complete_responses
        self.difficulty_and_discrimination_sha256 = result.difficulty_and_discrimination_sha256


def score_run(benchmark, benchmark_run, sut) -> SafetyScore:
    test_uids = {uid for hazard in benchmark.hazards() for uid in hazard.test_uids()}
    tests = [test for test in benchmark_run.tests if test.uid in test_uids]
    form_rows = [form_row(item).model_dump() for test in tests for item in test.make_test_items()]
    finished = benchmark_run.finished_items.get(sut.uid, {})
    evaluations = [
        evaluation_row(item, sut.uid, "Safety 1.2.1").model_dump()
        for test in tests
        for item in finished.get(test.uid, [])
    ]
    try:
        raw = science.score(benchmark.prompt_set, benchmark.locale, sut.uid, form_rows, evaluations)
    except SciencePublicationBlocked as blocked:
        raise PublicationBlocked(blocked.sut_uid, blocked.hazards, blocked.coverage) from blocked
    return SafetyScore(benchmark, sut, SafetyScoringResult.model_validate(raw), datetime.now())
