import os
import random
from typing import Dict, List, Optional
from tqdm import tqdm
from newhelm.annotation import Annotation
from newhelm.base_test import BasePromptResponseTest
from newhelm.cache_helper import SUTResponseCache
from newhelm.dependency_helper import FromSourceDependencyHelper
from newhelm.record_init import get_initialization_record
from newhelm.records import TestItemRecord, TestRecord
from newhelm.single_turn_prompt_response import (
    TestItemAnnotations,
    MeasuredTestItem,
    PromptInteraction,
    TestItemInteractions,
)
from newhelm.sut import PromptResponseSUT


def run_prompt_response_test(
    test_name: str,
    test: BasePromptResponseTest,
    sut_name: str,
    sut: PromptResponseSUT,
    data_dir: str,
    max_test_items: Optional[int] = None,
) -> TestRecord:
    """Demonstration for how to run a single Test on a single SUT, all calls serial."""

    # Ensure we can record what these objects are
    test_initialization = get_initialization_record(test)
    sut_initialization = get_initialization_record(sut)
    test_data_path = os.path.join(data_dir, test.get_metadata().name)

    # This runner just records versions, it doesn't specify a required version.
    dependency_helper = FromSourceDependencyHelper(
        test_data_path,
        test.get_dependencies(),
        required_versions={},
    )

    test_items = test.make_test_items(dependency_helper)
    if max_test_items and max_test_items < len(test_items):
        rng = random.Random()
        rng.seed(0)
        test_items = rng.sample(test_items, max_test_items)
    item_interactions: List[TestItemInteractions] = []
    desc = f"Collecting responses to {test_name} from {sut_name}"
    with SUTResponseCache(
        os.path.join(test_data_path, "cached_responses"), sut_name
    ) as cache:
        for item in tqdm(test_items, desc=desc):
            interactions = []
            for prompt in item.prompts:
                sut_request = sut.translate_request(prompt.prompt)
                sut_response = cache.get_cached_response(sut_request)
                if sut_response is None:
                    sut_response = sut.evaluate(sut_request)
                    cache.update_cache(sut_request, sut_response)
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
            try:
                annotation = annotator.annotate_test_item(
                    interactions_for_item.interactions
                )
            except Exception as e:
                raise Exception(
                    f"Exception while handling: {interactions_for_item}"
                ) from e
            annotations.append(Annotation.from_instance(annotation))
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
        test_name=test_name,
        test_initialization=test_initialization,
        dependency_versions=dependency_helper.versions_used(),
        sut_name=sut_name,
        sut_initialization=sut_initialization,
        test_item_records=test_item_records,
        results=results,
    )
