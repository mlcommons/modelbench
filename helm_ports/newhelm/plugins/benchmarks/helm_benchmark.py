from typing import Dict, List
from newhelm.base_test import BaseTest
from newhelm.benchmark import BaseBenchmark, Score
from newhelm.placeholders import Result
from newhelm.plugins.tests.bbq_imported import BBQImported


class HelmBenchmark(BaseBenchmark):
    def get_tests(self) -> List[BaseTest]:
        return [BBQImported()]

    def summarize(self, results: Dict[str, List[Result]]) -> Score:
        return Score(1.0)
