import os
import random
from typing import Dict, List, Optional

from tqdm import tqdm

from modelgauge.annotation import Annotation
from modelgauge.annotator import CompletionAnnotator
from modelgauge.base_test import PromptResponseTest, TestResult
from modelgauge.caching import Cache, NoCache, SqlDictCache
from modelgauge.dependency_helper import FromSourceDependencyHelper
from modelgauge.general import TestItemError
from modelgauge.prompt import TextPrompt
from modelgauge.records import TestItemExceptionRecord, TestItemRecord, TestRecord
from modelgauge.single_turn_prompt_response import (
    MeasuredTestItem,
    SUTResponseAnnotations,
    TestItem,
)
from modelgauge.sut import PromptResponseSUT
from modelgauge.sut_capabilities_verification import assert_sut_capabilities
from modelgauge.sut_decorator import assert_is_sut
from modelgauge.test_decorator import assert_is_test


def run_prompt_response_test(
    test: PromptResponseTest,
    sut: PromptResponseSUT,
    annotators: List[CompletionAnnotator],
    data_dir: str,
    max_test_items: Optional[int] = None,
    use_caching: bool = True,
    disable_progress_bar: bool = False,
) -> TestRecord:
    """Demonstration for how to run a single Test on a single SUT, all calls serial."""

    assert_is_test(test)
    assert_is_sut(sut)
    assert_sut_capabilities(sut, test)

    # Ensure we can record what these objects are
    test_initialization = test.initialization_record
    sut_initialization = sut.initialization_record
    # TODO: Add annotator initialization records
    test_data_path = os.path.join(data_dir, "tests", test.__class__.__name__)

    sut_cache: Cache
    if use_caching:
        sut_cache = SqlDictCache(os.path.join(data_dir, "suts"), sut.uid)
    else:
        sut_cache = NoCache()
    annotator_caches: dict[str, Cache] = {}
    for annotator in annotators:
        annotator_cache: Cache
        if use_caching:
            annotator_cache = SqlDictCache(os.path.join(test_data_path, "annotators"), annotator.uid)
        else:
            annotator_cache = NoCache()
        annotator_caches[annotator.uid] = annotator_cache

    # This runner just records versions, it doesn't specify a required version.
    dependency_helper = FromSourceDependencyHelper(
        os.path.join(test_data_path, "dependency_data"),
        test.get_dependencies(),
        required_versions={},
    )

    test_items = test.make_test_items(dependency_helper)
    if max_test_items is not None:
        assert max_test_items > 0, f"Cannot run a test using {max_test_items}."
        if max_test_items < len(test_items):
            rng = random.Random()
            rng.seed(0)
            rng.shuffle(test_items)
            test_items = test_items[:max_test_items]
    test_item_records = []
    test_item_exceptions = []
    measured_test_items = []
    desc = f"Processing TestItems for test={test.uid} sut={sut.uid}"
    for test_item in tqdm(test_items, desc=desc, disable=disable_progress_bar):
        try:
            test_item_record = _process_test_item(test_item, test, sut, sut_cache, annotators, annotator_caches)
            test_item_records.append(test_item_record)
            measured_test_items.append(
                MeasuredTestItem(
                    test_item=test_item_record.test_item,
                    measurements=test_item_record.measurements,
                )
            )
        except TestItemError as e:
            test_item_exceptions.append(
                TestItemExceptionRecord(test_item=test_item, error_message=str(e), cause=str(e.__cause__))
            )
    # TODO: Consider different threshold for num. items required to aggregate.
    if len(measured_test_items) > 0:
        test_result = TestResult.from_instance(test.aggregate_measurements(measured_test_items))
    else:
        test_result = TestResult.from_instance({"Unknown test result": "no valid items."})
    return TestRecord(
        test_uid=test.uid,
        test_initialization=test_initialization,
        sut_options=test.sut_options(),
        dependency_versions=dependency_helper.versions_used(),
        sut_uid=sut.uid,
        sut_initialization=sut_initialization,
        test_item_records=test_item_records,
        test_item_exceptions=test_item_exceptions,
        result=test_result,
    )


def _process_test_item(
    item: TestItem,
    test: PromptResponseTest,
    sut: PromptResponseSUT,
    sut_cache: Cache,
    annotators: List[CompletionAnnotator],
    annotator_caches: Dict[str, Cache],
) -> TestItemRecord:
    try:
        if isinstance(item.prompt, TextPrompt):
            sut_request = sut.translate_text_prompt(item.prompt, test.sut_options())
        else:
            sut_request = sut.translate_chat_prompt(item.prompt, test.sut_options())
        with sut_cache as cache:
            sut_response = cache.get_or_call(sut_request, sut.evaluate)
        response = sut.translate_response(sut_request, sut_response)
    except Exception as e:
        raise TestItemError(f"Exception while handling SUT {sut.uid} for prompt `{item.prompt}`") from e

    annotations = {}
    for annotator in annotators:
        try:
            with annotator_caches[annotator.uid] as cache:
                annotator_request = annotator.translate_request(item, response)
                annotator_response = cache.get_cached_response(annotator_request)
                if not annotator_response:
                    annotator_response = annotator.annotate(annotator_request)
                annotation = annotator.translate_response(annotator_request, annotator_response)
                cache.update_cache(annotator_request, annotator_response)
        except Exception as e:
            raise TestItemError(f"Exception while handling annotation for {annotator.uid} on `{response}`") from e

        annotations[annotator.uid] = Annotation.from_instance(annotation)
    annotated_response = SUTResponseAnnotations(test_item=item, sut_response=response, annotations=annotations)
    measurements = test.measure_quality(annotated_response)

    return TestItemRecord(
        test_item=annotated_response.test_item,
        sut_response_annotations=annotated_response,
        measurements=measurements,
    )
