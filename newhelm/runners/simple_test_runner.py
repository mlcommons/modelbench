import os
import random
from typing import List, Optional
from tqdm import tqdm
from newhelm.annotation import Annotation
from newhelm.base_annotator import CompletionAnnotator
from newhelm.base_test import BasePromptResponseTest, TestResult
from newhelm.caching import BaseCache, NoCache, SqlDictCache
from newhelm.dependency_helper import FromSourceDependencyHelper
from newhelm.prompt import TextPrompt
from newhelm.records import TestItemRecord, TestRecord
from newhelm.single_turn_prompt_response import (
    PromptInteractionAnnotations,
    SUTCompletionAnnotations,
    SUTResponseAnnotations,
    TestItem,
    TestItemAnnotations,
    MeasuredTestItem,
)
from newhelm.sut import PromptResponseSUT
from newhelm.sut_capabilities_verification import assert_sut_capabilities
from newhelm.sut_decorator import assert_is_sut
from newhelm.test_decorator import assert_is_test


def run_prompt_response_test(
    test: BasePromptResponseTest,
    sut: PromptResponseSUT,
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
    test_data_path = os.path.join(data_dir, test.__class__.__name__)

    sut_cache: BaseCache
    if use_caching:
        directory = os.path.join(test_data_path, "cached_responses")
        sut_cache = SqlDictCache(directory, sut.uid)
    else:
        sut_cache = NoCache()
    annotators = []
    for key, annotator in test.get_annotators().items():
        annotator_cache: BaseCache
        if use_caching:
            annotator_cache = SqlDictCache(
                os.path.join(test_data_path, "cached_annotations"), key
            )
        else:
            annotator_cache = NoCache()
        assert isinstance(
            annotator, CompletionAnnotator
        ), "Only know how to do CompletionAnnotator."
        annotators.append(AnnotatorData(key, annotator, annotator_cache))

    # This runner just records versions, it doesn't specify a required version.
    dependency_helper = FromSourceDependencyHelper(
        test_data_path,
        test.get_dependencies(),
        required_versions={},
    )

    test_items = test.make_test_items(dependency_helper)
    if max_test_items is not None:
        assert max_test_items > 0, f"Cannot run a test using {max_test_items}."
        if max_test_items < len(test_items):
            rng = random.Random()
            rng.seed(0)
            test_items = rng.sample(test_items, max_test_items)
    test_item_records = []
    measured_test_items = []
    desc = f"Processing TestItems for test={test.uid} sut={sut.uid}"
    for test_item in tqdm(test_items, desc=desc, disable=disable_progress_bar):
        test_item_record = _process_test_item(
            test_item, test, sut, sut_cache, annotators
        )
        test_item_records.append(test_item_record)
        measured_test_items.append(
            MeasuredTestItem(
                test_item=test_item_record.test_item,
                measurements=test_item_record.measurements,
            )
        )
    test_result = TestResult.from_instance(
        test.aggregate_measurements(measured_test_items)
    )
    return TestRecord(
        test_uid=test.uid,
        test_initialization=test_initialization,
        dependency_versions=dependency_helper.versions_used(),
        sut_uid=sut.uid,
        sut_initialization=sut_initialization,
        test_item_records=test_item_records,
        result=test_result,
    )


class AnnotatorData:
    """Container to hold data about an annotator."""

    def __init__(self, key: str, annotator: CompletionAnnotator, cache: BaseCache):
        self.key = key
        self.annotator = annotator
        self.cache = cache


def _process_test_item(
    item: TestItem,
    test: BasePromptResponseTest,
    sut: PromptResponseSUT,
    sut_cache: BaseCache,
    annotators: List[AnnotatorData],
) -> TestItemRecord:
    interactions: List[PromptInteractionAnnotations] = []
    for prompt in item.prompts:
        if isinstance(prompt.prompt, TextPrompt):
            sut_request = sut.translate_text_prompt(prompt.prompt)
        else:
            sut_request = sut.translate_chat_prompt(prompt.prompt)
        try:
            with sut_cache as cache:
                sut_response = cache.get_or_call(sut_request, sut.evaluate)
        except Exception as e:
            raise Exception(
                f"Exception while handling SUT request `{sut_request}` for TestItem `{item}`"
            ) from e
        response = sut.translate_response(sut_request, sut_response)

        annotated_completions: List[SUTCompletionAnnotations] = []
        for completion in response.completions:
            annotations = {}
            for annotator_data in annotators:
                annotator = annotator_data.annotator
                annotator_request = annotator.translate_request(prompt, completion)
                try:
                    with annotator_data.cache as cache:
                        annotator_response = cache.get_or_call(
                            annotator_request, annotator.annotate
                        )
                except Exception as e:
                    raise Exception(
                        f"Exception while handling annotation for {annotator_data.key} on {response}"
                    ) from e
                annotation = annotator.translate_response(
                    annotator_request, annotator_response
                )
                annotations[annotator_data.key] = Annotation.from_instance(annotation)
            annotated_completions.append(
                SUTCompletionAnnotations(completion=completion, annotations=annotations)
            )
        interactions.append(
            PromptInteractionAnnotations(
                prompt=prompt,
                response=SUTResponseAnnotations(completions=annotated_completions),
            )
        )
    annotated = TestItemAnnotations(
        test_item=item,
        interactions=interactions,
    )
    measurements = test.measure_quality(annotated)

    return TestItemRecord(
        test_item=annotated.test_item,
        interactions=annotated.interactions,
        measurements=measurements,
    )
