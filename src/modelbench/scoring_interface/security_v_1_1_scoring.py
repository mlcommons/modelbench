from datetime import datetime

from scoring_science import security_v_1_1_scoring as science
from scoring_science.security_v_1_1_scoring import (
    PAIR_FIELDS,
    PublicationBlocked as SciencePublicationBlocked,
)

from modelbench.benchmark_score import BaseBenchmarkScore
from modelbench.hazards import HazardScore, SecurityHazardV1_1
from modelbench.scoring import NoGradesMixin
from modelbench.scoring_interface.interface import (
    SecurityFormRow,
    PairedScoreBlocks,
    PublicationBlocked,
    SecurityScoringResult,
    evaluation_row,
)


class SecurityReportingHazard(SecurityHazardV1_1):
    def __init__(self, hazard_key, benchmark):
        super().__init__(benchmark.locale, benchmark.prompt_set)
        self.hazard_key = hazard_key

    @property
    def reference_key(self):
        return f"{super().reference_key}-{self.hazard_key}"

    _uid_definition = {
        **SecurityHazardV1_1._uid_definition,
        "hazard": "self.hazard_key",
    }


class SecurityHazardScore(NoGradesMixin, HazardScore):
    hazard: str
    score: PairedScoreBlocks

    def actual_score(self) -> float:
        return self.score.jailbreak.reference_percentile_score


class SecurityPairedScore(NoGradesMixin, BaseBenchmarkScore):
    def __init__(self, benchmark, sut, result: SecurityScoringResult, end_time):
        hazard_scores = [
            SecurityHazardScore(
                hazard_definition=SecurityReportingHazard(entry.hazard, benchmark),
                hazard=entry.hazard,
                score=PairedScoreBlocks(naive=entry.naive, jailbreak=entry.jailbreak),
                test_scores={},
                exceptions=0,
                num_scored_items=entry.num_scored_items,
            )
            for entry in result.hazard_scores
        ]
        super().__init__(benchmark, sut, hazard_scores, end_time)
        self.score = result.scores
        self.num_scored_items = result.complete_pairs
        self.difficulty_and_discrimination_sha256 = result.difficulty_and_discrimination_sha256


def form_row(item) -> SecurityFormRow:
    context = item.context if isinstance(item.context, dict) else {}
    if any(field not in context for field in PAIR_FIELDS):
        raise ValueError(f"Prompt {item.source_id} has no pairing fields; Security 1.1 needs the paired prompt file")
    return SecurityFormRow(prompt_uid=item.source_id, **{field: context[field] for field in PAIR_FIELDS})


def score_run(benchmark, benchmark_run, sut) -> SecurityPairedScore:
    test_uids = {uid for hazard in benchmark.hazards() for uid in hazard.test_uids()}
    tests = [test for test in benchmark_run.tests if test.uid in test_uids]
    form_rows = [form_row(item).model_dump() for test in tests for item in test.make_test_items()]
    finished = benchmark_run.finished_items.get(sut.uid, {})
    evaluations = [
        evaluation_row(item, sut.uid, "Security 1.1").model_dump()
        for test in tests
        for item in finished.get(test.uid, [])
    ]
    try:
        raw = science.score(benchmark.prompt_set, benchmark.locale, sut.uid, form_rows, evaluations)
    except SciencePublicationBlocked as blocked:
        raise PublicationBlocked(blocked.sut_uid, blocked.hazards, blocked.coverage) from blocked
    return SecurityPairedScore(benchmark, sut, SecurityScoringResult.model_validate(raw), datetime.now())
