from typing import Dict, List
from newhelm.plugins.tests.demo_02_unpacking_dependency_test import (
    DemoUnpackingDependencyTest,
)
from newhelm.plugins.tests.demo_03_paired_prompts_test import (
    DemoPairedPromptsTest,
)
from newhelm.aggregations import mean_of_results

from newhelm.plugins.tests.demo_01_simple_qa_test import DemoSimpleQATest
from newhelm.base_test import BaseTest
from newhelm.benchmark import BaseBenchmark, Score
from newhelm.placeholders import Result


class DemoBenchmark(BaseBenchmark):
    """A benchmark that runs all of the Demo Tests."""

    def get_tests(self) -> List[BaseTest]:
        """Returns the list of all Test objects this benchmark wants to run."""
        return [
            DemoSimpleQATest(),
            DemoUnpackingDependencyTest(),
            DemoPairedPromptsTest(),
        ]

    def summarize(self, results: Dict[str, List[Result]]) -> Score:
        """This demo reports the mean over all test Results."""
        return Score(mean_of_results(results))
