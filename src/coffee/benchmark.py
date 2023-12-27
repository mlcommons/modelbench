from abc import ABC, abstractmethod
from typing import List

import casefy

from coffee.helm import BbqHelmTest, HelmTest


class Benchmark(ABC):
    def __init__(self, sut, scores):
        super().__init__()
        self.sut = sut
        self.scores = scores

    @abstractmethod
    def overall_score(self) -> float:
        pass

    @abstractmethod
    def tests(self) -> List[HelmTest]:
        pass

    @classmethod
    def name(cls):
        return casefy.titlecase(cls.__name__)

    @classmethod
    def path_name(cls):
        return casefy.snakecase(cls.__name__)


class RidiculousBenchmark(Benchmark):
    def overall_score(self) -> float:
        bbq = self.scores["BbqHelmTest"]
        count = 0
        total = 0
        for subject in bbq:
            count += 1
            total += bbq[subject]["bbq_accuracy"]
        return total / count * 5

    @classmethod
    def tests(cls) -> List[HelmTest]:
        return [BbqHelmTest()]
