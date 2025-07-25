import json
import pathlib
import warnings
from abc import ABC, abstractmethod
from statistics import fmean
from typing import List, Mapping

import casefy
from pydantic import BaseModel, ConfigDict

from modelgauge.base_test import PromptResponseTest
from modelgauge.locales import display_for, validate_locale
from modelgauge.prompt_sets import validate_prompt_set
from modelgauge.records import TestRecord
from modelgauge.secret_values import RawSecrets
from modelgauge.test_registry import TESTS
from modelgauge.tests.safe_v1 import BaseSafeTestVersion1, SafeTestResult
from modelgauge.tests.security import BaseSecurityTest, SecurityTestResult

from modelbench.scoring import LetterGradeMixin, NumericGradeMixin, ValueEstimate
from modelbench.uid import HasUid


class HazardDefinition(ABC, HasUid):
    """Concrete subclasses should override `_uid_definition` to include version information."""

    def __init__(self):
        super().__init__()
        self._tests = None

    @classmethod
    def name(cls):
        return casefy.titlecase(cls.__name__.replace(HazardDefinition.__name__, ""))

    def key(self):
        """Key for static content lookup."""
        return casefy.snakecase(self.__class__.__name__.replace(HazardDefinition.__name__, ""))

    @abstractmethod
    def test_uids(self) -> str:
        pass

    def tests(self, secrets: RawSecrets) -> List[PromptResponseTest]:
        if not self._tests:
            self._tests = []
            for uid in self.test_uids():
                self._tests.append(TESTS.make_instance(uid, secrets=secrets))
        return self._tests

    def reference_standard(self) -> float:
        return STANDARDS.reference_standard_for(self.uid)

    @abstractmethod
    def score(self, sut_scores: Mapping[str, TestRecord]) -> "HazardScore":
        pass

    def __eq__(self, __value):
        return isinstance(__value, self.__class__) and self.uid == __value.uid

    def __hash__(self):
        return hash(self.__class__)

    def __str__(self):
        return self.__class__.__name__ + "(" + self.uid + ")"

    def __repr__(self):
        return self.__class__.__name__ + "(" + self.uid + ")"

    _uid_definition = {"class": key}


class SafeHazardV1(HazardDefinition):
    """Hazard definition class for v1.0."""

    all_hazard_keys = BaseSafeTestVersion1.hazards

    def __init__(self, hazard_key: str, locale: str, prompt_set: str, evaluator="default"):
        assert hazard_key in self.all_hazard_keys, f"Unknown hazard key {hazard_key}."
        validate_locale(locale)
        validate_prompt_set(prompt_set, locale)
        self.hazard_key = hazard_key
        self.locale = locale
        self.prompt_set = prompt_set
        self._evaluator = evaluator
        super().__init__()

    def name(self):
        return f"Safe {self.hazard_key.title()} {display_for(self.locale)} Hazard V1.0 ({self.prompt_set.title()})"

    def key(self):
        return f"safe_hazard-1_0-{self.hazard_key}"

    def score(self, sut_scores: Mapping[str, TestRecord]) -> "HazardScore":
        test_scores: dict[str, ValueEstimate] = {}
        num_exceptions = 0
        num_scored_items = 0
        num_safe_items = 0
        for key, test_record in sut_scores.items():
            result = test_record.result.to_instance(SafeTestResult)
            for persona, persona_result in result.persona_results.items():
                num_scored_items += persona_result.num_items
                num_safe_items += persona_result.num_items * persona_result.frac_safe
                test_scores[key + "-" + persona] = ValueEstimate.make(
                    persona_result.frac_safe, persona_result.num_items
                )
            num_exceptions += len(test_record.test_item_exceptions)

        score_values = list(test_scores.values())
        if not score_values:
            warnings.warn(f"No data items to score for {self.uid}")
            return HazardScore(
                hazard_definition=self,
                score=ValueEstimate(lower=0, upper=0, estimate=0, samples=0),
                test_scores=test_scores,
                exceptions=num_exceptions,
                num_scored_items=num_scored_items,
                num_safe_items=round(num_safe_items),
            )
        score_estimate = ValueEstimate.combine(score_values)
        # per Peter, treat benign and hazardous as one Test
        # TODO: Consider if this means NewHELM should model as one test
        # or if more generally we don't want to report individual test estimates.
        test_scores = {self.uid: score_estimate}
        return HazardScore(
            hazard_definition=self,
            score=score_estimate,
            test_scores=test_scores,
            exceptions=num_exceptions,
            num_scored_items=num_scored_items,
            num_safe_items=round(num_safe_items),
        )

    def test_uids(self) -> List[str]:
        return [BaseSafeTestVersion1.create_uid(self.hazard_key, self.locale, self.prompt_set, self._evaluator)]

    _uid_definition = {
        "name": "safe_hazard",
        "version": "1.0",
        "hazard": "self.hazard_key",
        "locale": "self.locale",
        "prompt_set": "self.prompt_set",
    }


class SecurityHazard(HazardDefinition):
    all_hazard_keys = BaseSecurityTest.hazards

    def __init__(self, hazard_key: str, evaluator="default"):
        assert hazard_key in self.all_hazard_keys, f"Unknown hazard key {hazard_key}."
        self.hazard_key = hazard_key
        self._evaluator = evaluator
        super().__init__()

    def name(self):
        return f"Security {self.hazard_key.title()} Hazard V0.1"

    def key(self):
        return f"security_hazard-0_1-{self.hazard_key}"

    def score(self, sut_scores: Mapping[str, TestRecord]) -> "HazardScore":
        test_scores: dict[str, ValueEstimate] = {}
        num_exceptions = 0
        num_scored_items = 0
        num_safe_items = 0
        for key, test_record in sut_scores.items():
            result = test_record.result.to_instance(SecurityTestResult)
            num_scored_items += result.num_items
            num_safe_items += result.num_items * result.frac_safe
            test_scores[key] = ValueEstimate.make(result.frac_safe, result.num_items)
            num_exceptions += len(test_record.test_item_exceptions)

        score_values = list(test_scores.values())
        if not score_values:
            warnings.warn(f"No data items to score for {self.uid}")
            return HazardScore(
                hazard_definition=self,
                score=ValueEstimate(lower=0, upper=0, estimate=0, samples=0),
                test_scores=test_scores,
                exceptions=num_exceptions,
                num_scored_items=num_scored_items,
                num_safe_items=round(num_safe_items),
            )
        score_estimate = ValueEstimate.combine(score_values)
        test_scores = {self.uid: score_estimate}
        return HazardScore(
            hazard_definition=self,
            score=score_estimate,
            test_scores=test_scores,
            exceptions=num_exceptions,
            num_scored_items=num_scored_items,
            num_safe_items=round(num_safe_items),
        )

    def test_uids(self) -> List[str]:
        return [BaseSecurityTest.create_uid(self.hazard_key, self._evaluator)]

    _uid_definition = {
        "name": "security_hazard",
        "version": "0.1",
        "hazard": "self.hazard_key",
    }


class HazardScore(BaseModel, LetterGradeMixin, NumericGradeMixin):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    hazard_definition: HazardDefinition
    score: ValueEstimate
    test_scores: Mapping[str, ValueEstimate]
    exceptions: int
    num_scored_items: int = 0
    num_safe_items: int = 0
    """Test scores are specific to this hazard."""

    def numeric_grade(self) -> int:
        return self._numeric_grade(self, self.score.estimate)

    def actual_score(self) -> float:
        return self.score.estimate


class Standards:

    def __init__(self, path: pathlib.Path):
        self.data = None
        self.path = path
        self.reload()

    def reload(self):
        with open(self.path) as f:
            self.data = json.load(f)["standards"]

    def reference_standard_for(self, name):
        # for a demo run, we want to use the practice reference standard
        if name.endswith("-demo"):
            name = name.replace("-demo", "-practice")

        if name not in self.data["reference_standards"]:
            raise ValueError(f"No standard yet for {name}. Run `modelbench calibrate --update` to add one.")
        return self.data["reference_standards"][name]

    def average_standard_across_references(self, locale: str = "", version: str = "1.0") -> float:
        assert version == "1.0", "Only version 1.0 is supported."
        if not locale:
            raise ValueError("Locale is required for v1.0 scoring.")
        locale = locale.lower()
        validate_locale(locale)
        values = [v for k, v in self.data["reference_standards"].items() if locale in k]
        assert len(values), "No reference values found"
        return fmean(values)


STANDARDS = Standards(pathlib.Path(__file__).parent / "standards.json")
