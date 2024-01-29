from typing import Dict, List
from newhelm.tests.bbq import BBQ

from newhelm.base_test import BaseTest
from newhelm.benchmark import BaseBenchmark, Score
from newhelm.placeholders import Result


class BBQBenchmark(BaseBenchmark):
    """This benchmark only exists to demonstrate using the BBQ Test."""

    def get_tests(self) -> List[BaseTest]:
        return [BBQ()]

    def summarize(self, results: Dict[str, List[Result]]) -> Score:
        """Given the results from each Test, produce a single Score."""
        # TODO make this real.
        return Score(value=0)
