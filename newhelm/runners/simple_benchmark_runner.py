import os
import random
from typing import Dict, List, Optional
from tqdm import tqdm
from newhelm.annotation import Annotation
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

    def __init__(self, data_dir: str, max_test_items: Optional[int] = None):
        self.data_dir = data_dir
        self.max_test_items = max_test_items

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
                test_records.append(
                    run_prompt_response_test(
                        test, sut, self.data_dir, self.max_test_items
                    )
                )
            # Run other kinds of tests on the SUT here
            test_results = {record.test_name: record.results for record in test_records}
            score = benchmark.summarize(test_results)
            benchmark_records.append(
                BenchmarkRecord(
                    benchmark_name=benchmark.__class__.__name__,
                    sut_name=sut.__class__.__name__,
                    test_records=test_records,
                    score=score,
                )
            )
        return benchmark_records


def run_prompt_response_test(
    test: BasePromptResponseTest,
    sut: PromptResponseSUT,
    data_dir: str,
    max_test_items: Optional[int] = None,
) -> TestRecord:
    """Demonstration for how to run a single Test on a single SUT, all calls serial."""
    # This runner just records versions, it doesn't specify a required version.
    dependency_helper = FromSourceDependencyHelper(
        os.path.join(data_dir, test.get_metadata().name),
        test.get_dependencies(),
        required_versions={},
    )

    test_items = test.make_test_items(dependency_helper)
    if max_test_items and max_test_items < len(test_items):
        random.seed(0)
        test_items = random.sample(test_items, max_test_items)
    item_interactions: List[TestItemInteractions] = []
    desc = f"Collecting responses to {test.__class__.__name__} from {sut.__class__.__name__}"
    for item in tqdm(test_items, desc=desc):
        interactions = []
        for prompt in item.prompts:
            sut_request = sut.translate_request(prompt.prompt)
            sut_response = sut.evaluate(sut_request)
            response = sut.translate_response(prompt.prompt, sut_response)
            interactions.append(PromptInteraction(prompt=prompt, response=response))
        item_interactions.append(
            TestItemInteractions(interactions=interactions, test_item=item)
        )
    annotations_per_annotator: Dict[str, List[Annotation]] = {}
    keyed_annotators = test.get_annotators().items()
    for key, annotator in keyed_annotators:
        annotations: List[Annotation] = []
        for interactions_for_item in item_interactions:
            annotations.append(
                Annotation.from_instance(
                    annotator.annotate_test_item(interactions_for_item.interactions)
                )
            )
        annotations_per_annotator[key] = annotations
    # Flatten annotations across annotators
    with_annotations = []
    for i, interactions_for_item in enumerate(item_interactions):
        test_item_annotations = {
            key: annotations_per_annotator[key][i] for key, _ in keyed_annotators
        }
        with_annotations.append(
            TestItemAnnotations(
                test_item=interactions_for_item.test_item,
                interactions=interactions_for_item.interactions,
                annotations=test_item_annotations,
            )
        )

    measured_test_items = []
    test_item_records = []
    for annotated in with_annotations:
        measurements = test.measure_quality(annotated)
        test_item_records.append(
            TestItemRecord(
                test_item=annotated.test_item,
                interactions=annotated.interactions,
                annotations=annotated.annotations,
                measurements=measurements,
            )
        )
        measured_test_items.append(
            MeasuredTestItem(test_item=annotated.test_item, measurements=measurements)
        )
    results = test.aggregate_measurements(measured_test_items)
    return TestRecord(
        test_name=test.__class__.__name__,
        dependency_versions=dependency_helper.versions_used(),
        sut_name=sut.__class__.__name__,
        test_item_records=test_item_records,
        results=results,
    )
