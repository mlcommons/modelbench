from abc import ABC, abstractmethod
from typing import List
from newhelm.benchmark import BaseBenchmark
from newhelm.records import BenchmarkRecord
from newhelm.sut import SUT


class BaseBenchmarkRunner(ABC):
    """This is the base class for all the different ways of collecting all data for a Benchmark.

    Runners can differ on how they handle parallelism, async, retries, etc.
    """

    @abstractmethod
    def run(self, benchmark: BaseBenchmark, suts: List[SUT]) -> List[BenchmarkRecord]:
        pass
