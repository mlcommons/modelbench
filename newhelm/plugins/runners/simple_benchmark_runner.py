import os
from typing import List
from newhelm.base_test import BasePromptResponseTest
from newhelm.benchmark import BaseBenchmark
from newhelm.benchmark_runner import BaseBenchmarkRunner
from newhelm.dependency_helper import FromSourceDependencyHelper
from newhelm.journal import BenchmarkJournal, TestJournal
from newhelm.single_turn_prompt_response import (
    AnnotatedTestItem,
    MeasuredTestItem,
    PromptInteraction,
    TestItemInteractions,
)
from newhelm.sut import SUT, PromptResponseSUT


class SimpleBenchmarkRunner(BaseBenchmarkRunner):
    """Demonstration of running a whole benchmark on a SUT, all calls serial."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def run(self, benchmark: BaseBenchmark, suts: List[SUT]) -> List[BenchmarkJournal]:
        # Not all runners can run all Test types, so validate up front
        prompt_response_tests: List[BasePromptResponseTest] = []
        for test in benchmark.get_tests():
            if isinstance(test, BasePromptResponseTest):
                prompt_response_tests.append(test)
            else:
                raise Exception("Runner can't handle test:", test.__class__.__name__)

        # Validate all SUTs can do the requested test types
        if prompt_response_tests:
            for sut in suts:
                assert isinstance(sut, PromptResponseSUT)

        # Actually run the tests
        benchmark_journals = []
        for sut in suts:
            test_journals = []
            for test in prompt_response_tests:
                assert isinstance(
                    sut, PromptResponseSUT
                )  # Redo the assert to make type checking happy.
                test_journals.append(self._run_prompt_response_test(test, sut))
            # Run other kinds of tests on the SUT here
            test_results = {
                journal.test_name: journal.results for journal in test_journals
            }
            score = benchmark.summarize(test_results)
            benchmark_journals.append(
                BenchmarkJournal(
                    benchmark.__class__.__name__,
                    sut.__class__.__name__,
                    test_journals,
                    score,
                )
            )
        return benchmark_journals

    def _run_prompt_response_test(
        self, test: BasePromptResponseTest, sut: PromptResponseSUT
    ) -> TestJournal:
        """Demonstration for how to run a single Test on a single SUT, all calls serial."""
        # This runner just records versions, it doesn't specify a required version.
        dependency_helper = FromSourceDependencyHelper(
            os.path.join(self.data_dir, test.get_metadata().name),
            test.get_dependencies(),
            required_versions={},
        )

        test_items = test.make_test_items(dependency_helper)
        item_interactions = []
        for item in test_items:
            interactions = []
            for prompt in item.prompts:
                response = sut.evaluate(prompt.prompt)
                interactions.append(PromptInteraction(prompt, response))
            item_interactions.append(TestItemInteractions(interactions, item))
        # Here is where an annotator would go
        with_annotations = [AnnotatedTestItem(item) for item in item_interactions]
        measured_test_items = []
        for annotated in with_annotations:
            measurements = test.measure_quality(annotated)
            measured_test_items.append(
                MeasuredTestItem(
                    annotated.item_with_interactions.test_item, measurements
                )
            )
        results = test.aggregate_measurements(measured_test_items)
        return TestJournal(
            test.__class__.__name__,
            dependency_helper.versions_used(),
            sut.__class__.__name__,
            with_annotations,
            results,
        )
