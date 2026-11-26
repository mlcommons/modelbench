from datetime import datetime
from typing import Sequence

from modelbench.benchmark_score import NullScore
from modelbench.benchmarks import GeneralPurposeAiChatBenchmarkV1
from modelbench.hazards import HazardDefinition
from modelgauge.constants import DEFAULT_EVALUATOR
from modelgauge.tests.ai_verify import AiVerifyHazard, AiVerifyTest


class AiVerifyBenchmark(GeneralPurposeAiChatBenchmarkV1):

    VERSION = "1.3"

    def __init__(self, locale: str, prompt_set: str, evaluator: str = DEFAULT_EVALUATOR):
        assert locale == "en_us"
        super().__init__(locale, prompt_set, evaluator)

    def reference_benchmark(self) -> "BenchmarkDefinition":
        return self

    @property
    def PROMPT_SETS(self) -> dict:
        return AiVerifyTest.PROMPT_SETS

    def score_benchmark_run(self, benchmark_run, sut):
        return NullScore(self, sut, datetime.now())

    def _make_hazards(self) -> Sequence[HazardDefinition]:
        return [
            AiVerifyHazard(hazard_key, self.locale, self.prompt_set, self.evaluator)
            for hazard_key in AiVerifyHazard.all_hazard_keys
        ]

    _uid_definition = {
        "class": "ai_verify_benchmark",
        "version": "self.VERSION",
        "locale": "self.locale",
        "prompt_set": "self.prompt_set",
        "evaluator": "self.evaluator",
    }
