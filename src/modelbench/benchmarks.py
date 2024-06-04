from abc import ABC
from datetime import datetime, timezone
from typing import List

import casefy
from modelgauge.sut_registry import SUTS

from modelbench.hazards import (
    HazardDefinition,
    HazardScore,
    SafeHazard,
)
from modelbench.modelgauge_runner import ModelGaugeSut

# from modelbench.records import RunRecord
from modelbench.scoring import LetterGradeMixin


class BenchmarkScore(ABC, LetterGradeMixin):
    def __init__(
        self,
        benchmark_definition: "BenchmarkDefinition",
        sut: ModelGaugeSut,
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

    def run_sut(self, max_instances, secrets, sut) -> BenchmarkScore:
        sut_instance = SUTS.make_instance(sut.key, secrets=secrets)

        hazard_scores = []
        for hazard in self.hazards():
            hazard_score = hazard.run_sut(max_instances, secrets, sut_instance)
            hazard_scores.append(hazard_score)

        benchmark_end_time = datetime.now(timezone.utc)
        sut_score = self.score(sut, hazard_scores, benchmark_end_time)
        return sut_score
        # record = RunRecord(
        #     benchmark_name=self.name(),
        #     sut_key=sut.key,
        #     benchmark_score=self.score(sut, hazard_records, benchmark_end_time).numeric_grade(),
        #     benchmark_grade="",
        #     hazard_records=hazard_records,
        #     end_time=benchmark_end_time,
        #     package_version="0.1.0",
        # )
        # return record

    def score(self, sut, hazard_scores, benchmark_end_time):
        return BenchmarkScore(self, sut, hazard_scores, benchmark_end_time)

    def __eq__(self, __value):
        if not isinstance(__value, self.__class__):
            return False
        if not len(self._hazards) == len(__value._hazards):
            return False
        for mine, theirs in zip(self._hazards, __value._hazards):
            if not mine == theirs:
                return False
        return True

    def __hash__(self):
        return hash((self.__class__, *self._hazards))


class GeneralPurposeAiChatBenchmark(BenchmarkDefinition):
    def __init__(self):
        super().__init__([c() for c in SafeHazard.__subclasses__()])
