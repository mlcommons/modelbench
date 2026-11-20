from typing import Sequence

from modelbench.benchmarks import GeneralPurposeAiChatBenchmarkV1
from modelbench.hazards import HazardDefinition, SafeHazardV1
from modelgauge.constants import DEFAULT_EVALUATOR
from modelgauge.tests.safe_v1 import BaseSafeTestVersion1


class AiVerifyTest(BaseSafeTestVersion1):
    VERSION = "1.3"
    hazards = ["vcr", "ncr", "cse", "spc", "hte", "ssh", "sxc"]

    PROMPT_SETS = {
        "official": {
            "en_sg": "ai_verify_benchmark-1.3-en_sg-official_prompt_set}",
        },
    }


class AiVerifyHazard(SafeHazardV1):
    VERSION = "1.3"
    TEST_CLASS = AiVerifyTest  # pyright: ignore
    all_hazard_keys = AiVerifyTest.hazards  # pyright: ignore


class AiVerifyBenchmark(GeneralPurposeAiChatBenchmarkV1):

    VERSION = "1.3"

    def __init__(self, locale: str, prompt_set: str, evaluator: str = DEFAULT_EVALUATOR):
        assert locale == "en_sg"
        super().__init__(locale, prompt_set, evaluator)

    @property
    def PROMPT_SETS(self) -> dict:
        return AiVerifyTest.PROMPT_SETS

    def _make_hazards(self) -> Sequence[HazardDefinition]:
        return [
            AiVerifyHazard(hazard_key, self.locale, self.prompt_set) for hazard_key in AiVerifyHazard.all_hazard_keys
        ]

    _uid_definition = {
        "class": "ai_verify_benchmark",
        "version": "self.VERSION",
        "locale": "self.locale",
        "prompt_set": "self.prompt_set",
        "evaluator": "self.evaluator",
    }
