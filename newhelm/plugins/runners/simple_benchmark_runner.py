import os
from typing import List
from tqdm import tqdm
from newhelm.base_test import BasePromptResponseTest
from newhelm.benchmark import BaseBenchmark
from newhelm.benchmark_runner import BaseBenchmarkRunner
from newhelm.dependency_helper import FromSourceDependencyHelper
from newhelm.records import BenchmarkRecord, TestItemRecord, TestRecord
from newhelm.single_turn_prompt_response import (
    TestItemAnnotations,
    MeasuredTestItem,
    PromptInteraction,
    TestItemInteractions,
)
from newhelm.sut import SUT, PromptResponseSUT


class SimpleBenchmarkRunner(BaseBenchmarkRunner):
    """Demonstration of running a whole benchmark on a SUT, all calls serial."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def run(self, benchmark: BaseBenchmark, suts: List[SUT]) -> List[BenchmarkRecord]:
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
        benchmark_records = []
        for sut in suts:
            test_records = []
            for test in prompt_response_tests:
                assert isinstance(
                    sut, PromptResponseSUT
                )  # Redo the assert to make type checking happy.
                test_records.append(self._run_prompt_response_test(test, sut))
            # Run other kinds of tests on the SUT here
            test_results = {record.test_name: record.results for record in test_records}
            score = benchmark.summarize(test_results)
            benchmark_records.append(
                BenchmarkRecord(
                    benchmark.__class__.__name__,
                    sut.__class__.__name__,
                    test_records,
                    score,
                )
            )
        return benchmark_records

    def _run_prompt_response_test(
        self, test: BasePromptResponseTest, sut: PromptResponseSUT
    ) -> TestRecord:
        """Demonstration for how to run a single Test on a single SUT, all calls serial."""
        # This runner just records versions, it doesn't specify a required version.
        dependency_helper = FromSourceDependencyHelper(
            os.path.join(self.data_dir, test.get_metadata().name),
            test.get_dependencies(),
            required_versions={},
        )

        test_items = test.make_test_items(dependency_helper)
        item_interactions: List[TestItemInteractions] = []
        desc = f"Collecting responses to {test.__class__.__name__} from {sut.__class__.__name__}"
        for item in tqdm(test_items, desc=desc):
            interactions = []
            for prompt in item.prompts:
                response = sut.evaluate(prompt.prompt)
                interactions.append(PromptInteraction(prompt, response))
            item_interactions.append(TestItemInteractions(interactions, item))
        # Here is where an annotator would go
        with_annotations = [
            TestItemAnnotations(item.test_item, item.interactions)
            for item in item_interactions
        ]
        measured_test_items = []
        test_item_records = []
        for annotated in with_annotations:
            measurements = test.measure_quality(annotated)
            test_item_records.append(
                TestItemRecord(
                    annotated.test_item,
                    annotated.interactions,
                    annotated.annotations,
                    measurements,
                )
            )
            measured_test_items.append(
                MeasuredTestItem(annotated.test_item, measurements)
            )
        results = test.aggregate_measurements(measured_test_items)
        return TestRecord(
            test.__class__.__name__,
            dependency_helper.versions_used(),
            sut.__class__.__name__,
            test_item_records,
            results,
        )
