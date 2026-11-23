from datetime import datetime
from typing import Sequence, List, Optional

from modelbench.benchmark_score import NullScore
from modelbench.benchmarks import GeneralPurposeAiChatBenchmarkV1
from modelbench.hazards import HazardDefinition, SafeHazardV1
from modelgauge.constants import DEFAULT_EVALUATOR
from modelgauge.locales import validate_locale
from modelgauge.personas import SafeTestPersonas
from modelgauge.prompt import TextPrompt
from modelgauge.prompt_sets import ModellabFileDownloadToken, validate_prompt_set
from modelgauge.secret_values import InjectSecret
from modelgauge.single_turn_prompt_response import TestItem
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.test_decorator import modelgauge_test
from modelgauge.test_registry import TESTS
from modelgauge.tests.safe_v1 import SafeTestVersion1_2_1


# this is copy-pasted from safety; it should be reconciled
class AiVerifyHazards(list):
    """The hazard keys are as follows:

    - vcr: Violent Crimes
    - ncr: Non-Violent Crimes
    - src: Sex-Related Crimes
    - cse: Child Sexual Exploitation
    - dfm: Defamation
    - spc: Specialized Advice
    - prv: Privacy Violations
    - ipv: Intellection Property Violations
    - iwp: Indiscriminate Weapons (CBRNE)
    - hte: Hate
    - ssh: Suicide & Self-Harm
    - sxc: Sexual Content"""

    SUB_HAZARDS = {
        "spc": {"spc_ele", "spc_fin", "spc_hlt", "spc_lgl"},
        "sxc": {"sxc_prn"},
    }

    def __init__(self):
        super().__init__(["vcr", "ncr", "cse", "spc", "hte", "ssh", "sxc"])

    @property
    def _known_hazards(self):
        return set(self).union(*self.SUB_HAZARDS.values())

    def get_hazard_family_from_row(self, row: dict[str, str]) -> str:
        """Returns the first half of a two-part hazard string (e.g. "spc_ele" -> "spc")
        or the whole hazard string as-is if there's no _ (e.g. "vcr" -> "vcr")"""
        return row["hazard"].split("_")[0]

    def get_hazard_from_row(self, row: dict[str, str]) -> str:
        """Returns a hazard string as-is if it's valid (e.g. "vcr", "spc_ele", etc)"""
        hazard = row["hazard"]
        if hazard not in self._known_hazards:
            raise ValueError(
                f"Unknown hazard code {hazard!r} for prompt {row.get('release_prompt_id', '<unknown>')!r}."
            )
        return hazard


@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
class AiVerifyTest(SafeTestVersion1_2_1):
    VERSION = "1.3"
    hazards = AiVerifyHazards()

    PROMPT_SETS = {
        "official": {
            "en_us": "ai_verify_benchmark-1.3-en_us-official_prompt_set",
        },
    }

    @classmethod
    def create_uid(cls, hazard: str, locale: str, prompt_set: str, evaluator=None):
        validate_prompt_set(cls.PROMPT_SETS, prompt_set, locale)
        validate_locale(locale)
        if evaluator is None or evaluator == DEFAULT_EVALUATOR:
            suffix = ""
        else:
            suffix = "-" + evaluator
        uid = f"ai_verify-{hazard}-{locale}-{prompt_set}-{cls.VERSION}{suffix}".lower()
        return uid

    def _test_item_for_row(self, row: dict) -> Optional[TestItem]:
        text = row["prompt_text"].strip()
        if not text:
            return None

        visibility = row.get("visibility", "private")
        context = self._test_item_context_for_row(row)
        return TestItem(
            prompt=TextPrompt(text=text),
            source_id=row["prompt_uid"],
            context=context,
            shareable=(visibility == "shareable"),
        )


# copy-paste ritual. What's the right solution?
def register_tests(test_class, evaluator=None):
    all_personas = [
        SafeTestPersonas.NORMAL,
        SafeTestPersonas.SKILLED,
        SafeTestPersonas.UNSKILLED,
    ]
    for prompt_set, prompt_set_files in test_class.PROMPT_SETS.items():
        for locale in prompt_set_files.keys():
            for hazard in test_class.hazards:
                test_uid = test_class.create_uid(hazard, locale, prompt_set, evaluator)
                token = InjectSecret(ModellabFileDownloadToken)
                TESTS.register(test_class, test_uid, hazard, locale, all_personas, prompt_set, token)


register_tests(AiVerifyTest, "private")


class AiVerifyHazard(SafeHazardV1):
    VERSION = "1.3"
    TEST_CLASS = AiVerifyTest
    all_hazard_keys = AiVerifyTest.hazards

    @property
    def reference_key(self):
        return f"ai_verify_hazard-{self.VERSION.replace('.', '_')}-{self.hazard_key}"

    _uid_definition = {
        "name": "ai_verify_hazard",
        "version": "self.VERSION",
        "hazard": "self.hazard_key",
        "locale": "self.locale",
        "prompt_set": "self.prompt_set",
    }


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
