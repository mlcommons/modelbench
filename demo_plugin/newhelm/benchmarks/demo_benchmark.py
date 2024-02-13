from typing import Dict, List, Mapping
from newhelm.benchmark_registry import BENCHMARKS
from newhelm.tests.demo_02_unpacking_dependency_test import (
    DemoUnpackingDependencyTest,
)
from newhelm.tests.demo_03_paired_prompts_test import (
    DemoPairedPromptsTest,
)
from newhelm.aggregations import mean_of_results

from newhelm.tests.demo_01_simple_qa_test import DemoSimpleQATest
from newhelm.base_test import BaseTest
from newhelm.benchmark import BaseBenchmark, Score
from newhelm.placeholders import Result


class DemoBenchmark(BaseBenchmark):
    """A benchmark that runs all of the Demo Tests."""

    def get_tests(self) -> Mapping[str, BaseTest]:
        """All of the demo Tests."""
        return {
            "demo_01": DemoSimpleQATest(),
            "demo_02": DemoUnpackingDependencyTest(),
            "demo_03": DemoPairedPromptsTest(),
        }

    def summarize(self, results: Dict[str, List[Result]]) -> Score:
        """This demo reports the mean over all test Results."""
        return Score(value=mean_of_results(results))


BENCHMARKS.register("demo", DemoBenchmark)
