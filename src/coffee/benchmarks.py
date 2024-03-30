from abc import ABC
from datetime import datetime, timedelta
from typing import List, Mapping, Union

import casefy
from newhelm.records import TestRecord

from coffee.hazards import (
    HazardDefinition,
    HazardScore,
    SafeHazard,
)
from coffee.newhelm_runner import NewhelmSut
from coffee.scoring import LetterGradeMixin, ValueEstimate


class BenchmarkScore(ABC, LetterGradeMixin):
    def __init__(
        self,
        benchmark_definition: "BenchmarkDefinition",
        sut: NewhelmSut,
        hazard_scores: List["HazardScore"],
        end_time: datetime,
    ):
        super().__init__()
        self.benchmark_definition = benchmark_definition
        self.sut = sut
        self.hazard_scores = hazard_scores
        self.end_time = end_time

    def numeric_grade(self):
        return min([hazard.numeric_grade() for hazard in self.hazard_scores])

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + str(self.benchmark_definition)
            + ", "
            + str(self.sut)
            + ", "
            + str(self.hazard_scores)
            + ")"
        )


class BenchmarkDefinition(ABC):
    _hazards: list[HazardDefinition]

    def __init__(self, hazards: List[HazardDefinition]):
        super().__init__()
        self._hazards = hazards

    def hazards(self) -> List[HazardDefinition]:
        return self._hazards

    @classmethod
    def name(cls):
        return casefy.titlecase(cls.__name__.replace(BenchmarkDefinition.__name__, ""))

    @classmethod
    def path_name(cls):
        return casefy.snakecase(cls.__name__.replace("Definition", ""))

    def score(self, sut, hazard_scores, benchmark_end_time):
        return BenchmarkScore(self, sut, hazard_scores, benchmark_end_time)


class GeneralPurposeAiChatBenchmark(BenchmarkDefinition):
    def __init__(self):
        super().__init__([c() for c in SafeHazard.__subclasses__()])
