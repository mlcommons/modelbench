from typing import List
from newhelm.annotation import AnnotatedInteraction
from newhelm.base_test import BasePromptResponseTest
from newhelm.benchmark import BaseBenchmark
from newhelm.benchmark_runner import BaseBenchmarkRunner
from newhelm.journal import BenchmarkJournal, TestJournal
from newhelm.sut import SUT, PromptResponseSUT


class SimpleBenchmarkRunner(BaseBenchmarkRunner):
    """Demonstration of running a whole benchmark on a SUT, all calls serial."""

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
        prompts = test.make_prompts()
        interactions = []
        for prompt in prompts:
            interactions.append(sut.evaluate(prompt))
        # Here is where an annotator would go
        annotated = [AnnotatedInteraction(interaction) for interaction in interactions]
        results = test.calculate_results(annotated)
        return TestJournal(
            test.__class__.__name__, sut.__class__.__name__, annotated, results
        )
