import importlib.metadata
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pydantic import BaseModel
from typing import List, Optional

import casefy

from modelbench.hazards import (
    HazardDefinition,
    HazardRun,
    SafeHazard,
)
from modelbench.record_init import record_init
from modelbench.scoring import LetterGradeMixin
from modelgauge.record_init import InitializationRecord


class BenchmarkScore(BaseModel):
    numeric_grade: int
    letter_grade: str


class BenchmarkRun(BaseModel):
    benchmark_init_record: InitializationRecord
    sut_init_record: InitializationRecord  # Maybe just use ModelGaugeSUT at this level
    benchmark_name: str
    max_instances: Optional[int]
    benchmark_score: BenchmarkScore
    hazard_runs: List[HazardRun]
    end_time: datetime
    package_version: str

    # # TODO: Extract to presentation layer, which gets these values from object initialization records.
    # benchmark_path_name: str
    # sut_uid: str

    # TODO: Taken from BenchmarkScore. Do we need this?
    # def __repr__(self):
    #     return (
    #         self.__class__.__name__
    #         + "("
    #         + str(self.benchmark_definition)
    #         + ", "
    #         + str(self.sut)
    #         + ", "
    #         + str(self.hazard_scores)
    #         + ")"
    #     )


class BenchmarkDefinition(ABC):
    @record_init
    def __init__(
        self,
        max_instances=None,
    ):
        super().__init__()
        self.max_instances = max_instances  # TODO: Use self.max_instances in methods
        # TODO: Keep secrets here

    #
    def run_sut(self, sut_instance, secrets, max_instances=None) -> BenchmarkRun:
        hazard_runs = self._run_hazards(sut_instance, secrets, max_instances)
        return self.build_run(sut_instance, hazard_runs, max_instances)

    def build_run(self, sut_instance, hazard_runs: List[HazardRun], max_instances=None) -> BenchmarkRun:
        # TODO: Assert hazard_records are all from this benchmark/SUT
        score = self._score(hazard_runs)
        benchmark_end_time = datetime.now(timezone.utc)
        return BenchmarkRun(
            benchmark_init_record=self.initialization_record,  # Set by record_init decorator
            sut_init_record=sut_instance.initialization_record,
            # benchmark_name=self.name(),
            # benchmark_path_name=self.path_name(),
            # sut_uid=sut_instance.uid,
            max_instances=max_instances,
            benchmark_score=score,
            hazard_runs=hazard_runs,
            end_time=benchmark_end_time,
            package_version=importlib.metadata.version("modelbench"),
        )

    def _run_hazards(self, sut_instance, secrets, max_instances=None) -> List[HazardRun]:
        hazard_runs = []
        for hazard in self.get_hazards():
            hazard_runs.append(hazard.run_sut(sut_instance, secrets, max_instances))
        return hazard_runs

    @abstractmethod
    def get_hazards(self) -> List[HazardDefinition]:
        pass

    @abstractmethod
    def _score(self, hazard_records: List[HazardRun]) -> BenchmarkScore:
        pass

    # I don't think we need this? Repeated in the TOML content
    @classmethod
    def name(cls):
        return casefy.titlecase(cls.__name__.replace(BenchmarkDefinition.__name__, ""))

    @classmethod
    def path_name(cls):
        return casefy.snakecase(cls.__name__.replace("Definition", ""))

    # TODO
    def __eq__(self, __value):
        if not isinstance(__value, self.__class__):
            return False
        hazards = self.get_hazards()
        if not len(hazards) == len(_value.get_hazards()):
            return False
        for mine, theirs in zip(hazards, __value.get_hazards()):
            if not mine == theirs:
                return False
        return True

    # TODO can we delete?
    def __hash__(self):
        return hash((self.__class__, *self.get_hazards()))


class GeneralPurposeAiChatBenchmark(BenchmarkDefinition, LetterGradeMixin):
    @record_init
    def __init__(self, max_instances=None):
        super().__init__(max_instances)

    def get_hazards(self) -> List[HazardDefinition]:
        return [c() for c in SafeHazard.__subclasses__()]

    def _score(self, hazard_runs: List[HazardRun]) -> BenchmarkScore:
        return BenchmarkScore(
            numeric_grade=self._numeric_grade(hazard_runs),
            letter_grade=self._letter_grade(hazard_runs),
        )

    def _numeric_grade(self, hazard_runs: List[HazardRun]) -> int:
        return min([run.hazard_score.numeric_grade for run in hazard_runs])
