from typing import Dict, List

from newhelm.plugins.tests.mmlu import MMLU
from newhelm.base_test import BaseTest
from newhelm.benchmark import BaseBenchmark, Score
from newhelm.placeholders import Result
from newhelm.plugins.tests.bbq import BBQ


class RidiculousBenchmark(BaseBenchmark):
    def get_tests(self) -> List[BaseTest]:
        return [BBQ(), MMLU()]

    def summarize(self, results: Dict[str, List[Result]]) -> Score:
        """Given the results from each Test, produce a single Score."""
        # In a real implementation, this would probably use a library function.
        flattened = sum(results.values(), [])
        if len(flattened) == 0:
            return Score(value=0)
        total = sum(r.value for r in flattened)
        return Score(value=total / len(flattened))
