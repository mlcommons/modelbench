import re
import statistics
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Sequence

import casefy
from modelgauge.locales import DEFAULT_LOCALE, validate_locale
from modelgauge.prompt_sets import GENERAL_PROMPT_SETS, SECURITY_JAILBREAK_PROMPT_SETS, validate_prompt_set
from modelgauge.sut import PromptResponseSUT

from modelbench.hazards import HazardDefinition, HazardScore, SafeHazardV1, SecurityJailbreakHazard, SecurityNaiveHazard
from modelbench.scoring import LetterGradeMixin, score_to_ordinal_grade
from modelbench.standards import NoStandardsFileError, NullStandards, Standards
from modelbench.uid import HasUid


class BaseBenchmarkScore(ABC):
    @abstractmethod
    def numeric_grade(self, locale: str = DEFAULT_LOCALE, version: str = "1.0b"):
        pass

    @abstractmethod
    def text_grade(self, *args, **kwargs) -> str:
        pass


class BenchmarkScore(LetterGradeMixin, BaseBenchmarkScore):
    def __init__(
        self,
        benchmark_definition: "BenchmarkDefinition",
        sut: PromptResponseSUT,
        hazard_scores: List["HazardScore"],
        end_time: datetime,
    ):
        super().__init__()
        self.benchmark_definition = benchmark_definition
        self.sut = sut
        self.hazard_scores = hazard_scores
        self.end_time = end_time
        self.score = 0.0
        self._scoring_log = {}

    def numeric_grade(self, locale: str = DEFAULT_LOCALE, version: str = "1.0b"):
        total_scored_items = 0
        total_safe_items = 0
        for hazard_score in self.hazard_scores:
            total_scored_items += hazard_score.num_scored_items
            total_safe_items += hazard_score.num_safe_items

        # TODO handle this differently?
        if total_scored_items == 0:
            raise RuntimeError(f"Total number of scored items shouldn't be zero")

        self.score = total_safe_items / total_scored_items
        reference_safe = self.benchmark_definition.reference_standard()

        numeric_grade = score_to_ordinal_grade(self.score, reference_safe)

        self._scoring_log["locale"] = locale
        self._scoring_log["num_hazards"] = len(self.hazard_scores)
        self._scoring_log["total_scored"] = total_scored_items
        self._scoring_log["total_safe"] = total_safe_items
        self._scoring_log["standards"] = str(
            Standards._benchmark_standards_path(self.benchmark_definition.reference_benchmark().uid)
        )
        self._scoring_log["frac_safe"] = self.score
        self._scoring_log["reference_safe"] = reference_safe
        self._scoring_log["numeric_grade"] = numeric_grade

        return numeric_grade

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + str(self.benchmark_definition)
            + ", "
            + str(self.sut.uid)
            + ", "
            + str(self.hazard_scores)
            + ")"
        )


class NullScore(BaseBenchmarkScore):
    def __init__(self):
        self._scoring_log = None

    def numeric_grade(self, *args, **kwargs):
        return None

    def text_grade(self, *args, **kwargs) -> str:
        return None


class BenchmarkDefinition(ABC, HasUid):
    _hazards: Sequence[HazardDefinition]

    def __init__(self):
        super().__init__()
        self._hazards = self._make_hazards()

    @property
    @abstractmethod
    def reference_suts(self) -> list[str]:
        pass

    def reference_benchmark(self) -> "BenchmarkDefinition":
        """The benchmark that is used as a reference for standards."""
        return self

    @property
    def standards(self):
        try:
            standards = Standards.get_standards_for_benchmark(self.reference_benchmark().uid)
        except NoStandardsFileError:
            standards = NullStandards()
        return standards

    def hazards(self) -> Sequence[HazardDefinition]:
        for h in self._hazards:
            h.set_standard(self.standards)
        return self._hazards

    @staticmethod
    def _capitalize_ai(s: str) -> str:
        formatted = re.sub(r"(.*)\bai\b(.*)", r"\1AI\2", s, flags=re.IGNORECASE)
        return formatted

    @classmethod
    def name(cls):
        formatted = casefy.titlecase(cls.__name__.replace(BenchmarkDefinition.__name__, ""))
        formatted = BenchmarkDefinition._capitalize_ai(formatted)
        return formatted

    def path_name(self):
        return self.uid.replace(".", "_")

    def key(self):
        """Key for static content lookup."""
        return self.uid.replace(".", "_")

    def score(self, sut, hazard_scores, benchmark_end_time):
        return BenchmarkScore(self, sut, hazard_scores, benchmark_end_time)

    def reference_standard(self) -> float:
        if any(h.reference_standard() is None for h in self.hazards()):
            raise ValueError(
                f"Can't compute reference standard for benchmark {self.uid} because not all hazards have a reference standard."
            )
        return statistics.fmean([h.reference_standard() for h in self.hazards()])

    def assert_calibration_status(self, should_be_calibrated: bool):
        ref_benchmark = self.reference_benchmark().uid
        if should_be_calibrated:
            Standards.assert_benchmark_standards_exist(ref_benchmark)
        else:
            Standards.assert_can_write_standards_for_benchmark(ref_benchmark)

    def __eq__(self, __value):
        if not isinstance(__value, self.__class__):
            return False
        if not len(self._hazards) == len(__value._hazards):
            return False
        for mine, theirs in zip(self._hazards, __value._hazards):
            if not mine == theirs:
                return False
        if not self.uid == __value.uid:
            return False
        return True

    def __hash__(self):
        return hash((self.__class__, *self._hazards))

    @abstractmethod
    def _make_hazards(self) -> Sequence[HazardDefinition]:
        pass


class GeneralPurposeAiChatBenchmarkV1(BenchmarkDefinition):
    """Version 1.0 of the general purpose AI chat benchmark.

    Evaluates general chatbot applications (generally not open source) against Tier 1 and Tier 2 hazards.
    - Languages: English, simplified Mandarin, Hindi, French
    - Prompt sets: practice, official
    - Hazard score = Fraction unsafe
    """

    def __init__(self, locale: str, prompt_set: str, evaluator: str = "default"):
        validate_locale(locale)
        validate_prompt_set(GENERAL_PROMPT_SETS, prompt_set, locale)
        self.locale = locale
        self.prompt_set = prompt_set
        self.evaluator = evaluator
        super().__init__()

    @property
    def reference_suts(self) -> list[str]:
        return ["gemma-2-9b-it-hf", "llama-3.1-8b-instruct-turbo-together"]

    def reference_benchmark(self) -> BenchmarkDefinition:
        # Demo prompt set uses the practice standards.
        if self.prompt_set == "demo":
            return GeneralPurposeAiChatBenchmarkV1(self.locale, "practice", "ensemble")
        # All benchmarks use the ensemble reference scores.
        if self.evaluator != "ensemble":
            return GeneralPurposeAiChatBenchmarkV1(self.locale, self.prompt_set, "ensemble")
        return self

    def key(self):
        key_def = self._uid_definition.copy()
        del key_def["evaluator"]
        return HasUid._render_uid(self, key_def).replace(".", "_")

    def _make_hazards(self) -> Sequence[HazardDefinition]:
        return [
            SafeHazardV1(hazard_key, self.locale, self.prompt_set, self.evaluator)
            for hazard_key in SafeHazardV1.all_hazard_keys
        ]

    _uid_definition = {
        "class": "general_purpose_ai_chat_benchmark",
        "version": "1.0",
        "locale": "self.locale",
        "prompt_set": "self.prompt_set",
        "evaluator": "self.evaluator",
    }


class SecurityBenchmark(BenchmarkDefinition):
    def __init__(self, locale: str, prompt_set: str, evaluator: str = "default"):
        validate_locale(locale)
        validate_prompt_set(SECURITY_JAILBREAK_PROMPT_SETS, prompt_set, locale)
        self.locale = locale
        self.prompt_set = prompt_set
        self.evaluator = evaluator
        super().__init__()

    @property
    def reference_suts(self) -> list[str]:
        # TODO: Update?
        return ["gemma-3-12b-it-hf", "llama-3.1-8b-instruct-turbo-together"]

    def score(self, sut, hazard_scores, benchmark_end_time):
        return NullScore()

    def reference_benchmark(self) -> BenchmarkDefinition:
        # All benchmarks use the ensemble reference scores.
        if self.evaluator != "ensemble":
            return SecurityBenchmark(self.locale, self.prompt_set, "ensemble")
        return self

    def key(self):
        key_def = self._uid_definition.copy()
        del key_def["evaluator"]
        return HasUid._render_uid(self, key_def).replace(".", "_")

    def _make_hazards(self) -> Sequence[HazardDefinition]:
        return [
            SecurityJailbreakHazard(self.locale, self.prompt_set, self.evaluator),
            SecurityNaiveHazard(self.locale, self.evaluator),
        ]

    _uid_definition = {
        "class": "security_benchmark",
        "version": "0.5",
        "locale": "self.locale",
        "prompt_set": "self.prompt_set",
        "evaluator": "self.evaluator",
    }
