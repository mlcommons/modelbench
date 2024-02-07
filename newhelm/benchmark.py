from abc import ABC, abstractmethod
from typing import Dict, List

from pydantic import BaseModel

from newhelm.base_test import BaseTest
from newhelm.placeholders import Result


class Score(BaseModel):
    # TODO figure out a clever way to ensure this is always at most half stars.
    value: float


class BaseBenchmark(ABC):
    """The base for all benchmarks."""

    @abstractmethod
    def get_tests(self) -> List[BaseTest]:
        """Return a list of tests that compose this Benchmark."""
        pass

    @abstractmethod
    def summarize(self, results: Dict[str, List[Result]]) -> Score:
        """Given the results from each Test, produce a single Score."""
        pass
