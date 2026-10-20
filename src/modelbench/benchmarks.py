import re
import statistics
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterator, Sequence

import casefy

from modelbench.benchmark_runner_items import TestRunItem
from modelbench.benchmark_score import BenchmarkScore, SecurityScore
from modelbench.external_scoring.scoring_adapters import RegressionScorer
from modelbench.hazards import (
    HazardDefinition,
    HazardScore,
    SafeHazardV1_1,
    SafeHazardV1_2,
    SecurityJailbreakHazardV1_0_2,
    SecurityJailbreakHazardV1_1,
    SecurityNaiveHazardV1_0_2,
)
from modelbench.standards import NoStandardsFileError, NullStandards, Standards
from modelbench.uid import HasUid
from modelgauge.locales import validate_locale
from modelgauge.prompt_sets import (
    GENERAL_PROMPT_SETS,
    SAFETY_1_2_PROMPT_SETS,
    SECURITY_1_0_1_JAILBREAK_PROMPT_SETS,
    SECURITY_1_1_JAILBREAK_PROMPT_SETS,
    validate_prompt_set,
)
from modelgauge.versioned_object import VersionedObject


class BenchmarkDefinition(VersionedObject, HasUid, ABC):
    _hazards: Sequence[HazardDefinition]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

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

    def score(self, sut, hazard_scores, benchmark_end_time):
        return BenchmarkScore(self, sut, hazard_scores, benchmark_end_time)

    def score_benchmark_run(self, benchmark_run: "BenchmarkRun", sut):
        hazard_scores = self._score_hazards(benchmark_run, sut.uid)
        return self.score(sut, hazard_scores, datetime.now())

    def _score_hazards(self, benchmark_run: "BenchmarkRun", sut_uid) -> list[HazardScore]:
        hazard_scores = []
        for hazard in self.hazards():
            hazard_score = self._score_hazard(benchmark_run, hazard, sut_uid)
            hazard_scores.append(hazard_score)
        return hazard_scores

    def _score_hazard(self, benchmark_run: "BenchmarkRun", hazard: HazardDefinition, sut_uid: str) -> HazardScore:
        test_records = {}
        for test in hazard.tests(benchmark_run.secrets):
            records = benchmark_run.test_records[test.uid][sut_uid]
            assert records, f"No records found for {self} {sut_uid} {hazard} {test.uid}"
            test_records[test.uid] = records

        assert test_records, f"No records found for {self} {sut_uid} {hazard}"

        return hazard.score(test_records)

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
            Standards.assert_file_does_not_exist(ref_benchmark)

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


class GeneralPurposeAiChatBenchmarkV1(BenchmarkDefinition, ABC):
    """Common functionality for the general-purpose AI chat benchmark.
    Different versions should be subclasses of this, extending
    `modelgauge.versioned_object.VersionedObject`, and have values
    for VERSION (string, e.g. "1.3") and PROMPT_SETS (a dict, from modelgauge.prompt_sets).

    Evaluates general chatbot applications (generally not open source) against Tier 1 and Tier 2 hazards.
    - Languages: English, simplified Mandarin, Hindi, French
    - Prompt sets: practice, official
    - Hazard score = Fraction unsafe
    """

    @property
    @abstractmethod
    def PROMPT_SETS(self) -> dict:
        pass

    def __init__(self, locale: str, prompt_set: str, evaluator: str = "default"):
        validate_locale(locale)
        validate_prompt_set(self.PROMPT_SETS, prompt_set, locale)
        self.locale = locale
        self.prompt_set = prompt_set
        self.evaluator = evaluator
        super().__init__()

    @property
    def reference_suts(self) -> list[str]:
        return ["llama-3.1-8b-instruct-turbo-together"]

    def _make_hazards(self, hazard_class: type[HazardDefinition]) -> Sequence[HazardDefinition]:
        return [
            hazard_class(hazard_key, self.locale, self.prompt_set, self.evaluator)
            for hazard_key in hazard_class.all_hazard_keys
        ]

    def reference_benchmark(self) -> BenchmarkDefinition:
        # Demo prompt set uses the practice standards.
        if self.prompt_set == "demo":
            return type(self)(self.locale, "practice", "private")
        # All benchmarks use the private reference scores.
        if self.evaluator != "private":
            return type(self)(self.locale, self.prompt_set, "private")
        return self

    _uid_definition = {
        "class": "general_purpose_ai_chat_benchmark",
        "version": "self.VERSION",
        "locale": "self.locale",
        "prompt_set": "self.prompt_set",
        "evaluator": "self.evaluator",
    }


class GeneralPurposeAiChatBenchmarkV1_1(GeneralPurposeAiChatBenchmarkV1):
    VERSION = "1.1"
    PROMPT_SETS = GENERAL_PROMPT_SETS  # pyright: ignore

    def _make_hazards(self) -> Sequence[HazardDefinition]:
        return super()._make_hazards(SafeHazardV1_1)


class GeneralPurposeAiChatBenchmarkV1_2(GeneralPurposeAiChatBenchmarkV1):
    VERSION = "1.2"
    PROMPT_SETS = SAFETY_1_2_PROMPT_SETS  # pyright: ignore

    def _make_hazards(self) -> Sequence[HazardDefinition]:
        return super()._make_hazards(SafeHazardV1_2)

    def score_benchmark_run(self, benchmark_run: "BenchmarkRun", sut) -> BenchmarkScore:
        assert len(benchmark_run.benchmarks) == 1
        the_benchmark = benchmark_run.benchmarks[0]
        assert isinstance(the_benchmark, self.__class__)
        items: list[TestRunItem] = []
        for sut_uid in benchmark_run.finished_items:
            for test_uid in benchmark_run.finished_items[sut_uid]:
                items.extend(benchmark_run.finished_items[sut_uid][test_uid])

        scorer = RegressionScorer()
        return scorer.score(the_benchmark, sut, items)


class NaiveBenchmarkV1_0_2(GeneralPurposeAiChatBenchmarkV1):
    VERSION = "1.0.2"
    PROMPT_SETS = GENERAL_PROMPT_SETS  # pyright: ignore

    @property
    def reference_suts(self) -> list[str]:
        return ["llama-3.1-8b-instruct-turbo-together"]

    def score(self, sut, hazard_scores, benchmark_end_time):
        return SecurityScore(self, sut, hazard_scores, benchmark_end_time)

    def reference_benchmark(self) -> BenchmarkDefinition:
        return self

    def _make_hazards(self) -> Sequence[HazardDefinition]:
        return [SecurityNaiveHazardV1_0_2(self.locale, self.prompt_set, self.evaluator)]

    _uid_definition = {
        "class": "security_naive_benchmark",
        "version": "self.VERSION",
        "locale": "self.locale",
        "prompt_set": "self.prompt_set",
        "evaluator": "self.evaluator",
    }


class SecurityBenchmarkV1_0_2(GeneralPurposeAiChatBenchmarkV1):
    VERSION = "1.0.2"
    PROMPT_SETS = SECURITY_1_0_1_JAILBREAK_PROMPT_SETS  # pyright: ignore

    @property
    def reference_suts(self) -> list[str]:
        return self.reference_benchmark().reference_suts

    def score(self, sut, hazard_scores, benchmark_end_time):
        return SecurityScore(self, sut, hazard_scores, benchmark_end_time)

    def reference_benchmark(self) -> BenchmarkDefinition:
        return NaiveBenchmarkV1_0_2(self.locale, "official", "private")

    def _make_hazards(self) -> Sequence[HazardDefinition]:
        return [
            SecurityJailbreakHazardV1_0_2(self.locale, self.prompt_set, self.evaluator),
            SecurityNaiveHazardV1_0_2(self.locale, self.prompt_set, self.evaluator),
        ]

    _uid_definition = {
        "class": "security_benchmark",
        "version": "self.VERSION",
        "locale": "self.locale",
        "prompt_set": "self.prompt_set",
        "evaluator": "self.evaluator",
    }


class SecurityBenchmarkV1_1(GeneralPurposeAiChatBenchmarkV1):
    VERSION = "1.1"
    PROMPT_SETS = SECURITY_1_1_JAILBREAK_PROMPT_SETS  # pyright: ignore

    def score_benchmark_run(self, benchmark_run, sut):
        from modelbench.benchmark_score import SecurityPairedScore
        from modelbench.external_scoring.security_v_1_1_scoring import score_stacked_sut
        import hashlib
        import json
        from pathlib import Path

        test_uids = {uid for hazard in self.hazards() for uid in hazard.test_uids()}
        wrappers = [test for test in benchmark_run.tests if test.uid in test_uids]
        expected_items = [item for test in wrappers for item in test.make_test_items()]
        if not any(isinstance(item.context, dict) and "prompt_condition" in item.context for item in expected_items):
            return super().score_benchmark_run(benchmark_run, sut)
        form_rows = []
        for item in expected_items:
            if not isinstance(item.context, dict) or "prompt_condition" not in item.context:
                raise ValueError("Mixed stacked and legacy Security prompt sets")
            form_rows.append({"prompt_uid": item.source_id, **item.context})

        path = getattr(self, "scoring_config_path", None)
        if not path:
            raise ValueError("Stacked Security requires an explicit frozen --scoring-config")
        payload = Path(path).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        expected_digest = getattr(self, "scoring_config_sha256", None)
        if expected_digest and digest != expected_digest:
            raise ValueError("Security scoring configuration SHA-256 does not match")
        config = json.loads(payload)
        evaluations = []
        finished = benchmark_run.finished_items.get(sut.uid, {})
        for test in wrappers:
            for item in finished.get(test.uid, []):
                if item.sut.uid != sut.uid:
                    raise ValueError("Finished Security item belongs to a different SUT")
                if len(item.annotations) != 1:
                    raise ValueError("Stacked Security requires one evaluator annotation per response")
                annotation = next(iter(item.annotations.values()))
                if not hasattr(annotation, "violation_probability"):
                    raise ValueError("Stacked Security requires continuous evaluator probabilities")
                evaluations.append(
                    {
                        "sut_uid": sut.uid,
                        "prompt_uid": item.source_id(),
                        "is_valid": annotation.is_valid,
                        "violation_probability": annotation.violation_probability,
                    }
                )
        paired = score_stacked_sut(config, sut.uid, form_rows, evaluations)
        paired["scoring_config_sha256"] = digest
        return SecurityPairedScore(self, sut, paired, datetime.now())

    def score(self, sut, hazard_scores, benchmark_end_time):
        return SecurityScore(self, sut, hazard_scores, benchmark_end_time)

    def _make_hazards(self) -> Sequence[HazardDefinition]:
        return [
            SecurityJailbreakHazardV1_1(self.locale, self.prompt_set, self.evaluator),
        ]

    _uid_definition = {
        "class": "security_benchmark",
        "version": "self.VERSION",
        "locale": "self.locale",
        "prompt_set": "self.prompt_set",
        "evaluator": "self.evaluator",
    }


def _walk_subclasses(cls: type) -> Iterator[type]:
    for sub in cls.__subclasses__():
        yield sub
        yield from _walk_subclasses(sub)


def benchmark_class_for(prefix: str, version: str) -> type[BenchmarkDefinition]:
    for cls in _walk_subclasses(BenchmarkDefinition):
        if cls.__name__.startswith(prefix) and getattr(cls, "VERSION", None) == version:
            return cls
    raise KeyError(f"No BenchmarkDefinition subclass with prefix '{prefix}' and VERSION '{version}'")


def _version_sort_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def benchmark_versions_for(prefix: str) -> list[str]:
    versions = {
        getattr(cls, "VERSION")
        for cls in _walk_subclasses(BenchmarkDefinition)
        if cls.__name__.startswith(prefix) and hasattr(cls, "VERSION")
    }
    return sorted(versions, key=_version_sort_key, reverse=True)
