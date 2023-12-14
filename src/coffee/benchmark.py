from abc import ABC, abstractmethod


class Benchmark(ABC):
    def __init__(self, sut, scores):
        super().__init__()
        self.sut = sut
        self.scores = scores

    @abstractmethod
    def overall_score(self) -> float:
        pass


class RidiculousBenchmark(Benchmark):
    def overall_score(self) -> float:
        bbq = self.scores["BbqHelmTest"]
        count = 0
        total = 0
        for subject in bbq:
            count += 1
            total += bbq[subject]["bbq_accuracy"]
        return total / count * 5
