import os
import pytest
from unittest import mock


from modelgauge.caching import SqlDictCache
from modelgauge.annotation import Annotation
from modelgauge.records import TestItemExceptionRecord, TestItemRecord
from modelgauge.simple_test_runner import run_prompt_response_test
from modelgauge.single_turn_prompt_response import (
    PromptInteractionAnnotations,
    SUTCompletionAnnotations,
    SUTResponseAnnotations,
)
from modelgauge.sut import SUTCompletion
from modelgauge.sut_capabilities import ProducesPerTokenLogProbabilities
from modelgauge.test_decorator import modelgauge_test
from tests.fake_annotator import FakeAnnotator
from tests.fake_sut import FakeSUT
from tests.fake_test import FakeTest, FakeTestResult, fake_test_item

_FAKE_MEASUREMENT = {"some-measurement": 0.5}


def _make_test_item_record(item):
    text = item.prompts[0].prompt.text

    return TestItemRecord(
        test_item=item,
        interactions=[
            PromptInteractionAnnotations(
                prompt=item.prompts[0],
                response=SUTResponseAnnotations(
                    completions=[
                        SUTCompletionAnnotations(
                            completion=SUTCompletion(text=text),
                            annotations={
                                "some-annotator": Annotation(
                                    module="tests.fake_annotator",
                                    class_name="FakeAnnotation",
                                    data={"sut_text": text},
                                )
                            },
                        )
                    ]
                ),
            )
        ],
        measurements=_FAKE_MEASUREMENT,
    )


def _make_sut_exception_record(item):
    return TestItemExceptionRecord(
        test_item=item,
        error_message=f"Exception while handling SUT fake-sut for prompt `{item.prompts[0]}`",
        cause="some-exception",
    )


def _make_annotator_exception_record(item):
    prompt_text = item.prompts[0].prompt.text
    return TestItemExceptionRecord(
        test_item=item,
        error_message=f"Exception while handling annotation for some-annotator on `{SUTCompletion(text=prompt_text)}`",
        cause="some-exception",
    )


def test_run_prompt_response_test_output(tmpdir):
    item_1 = fake_test_item("1")
    item_2 = fake_test_item("2")
    record = run_prompt_response_test(
        FakeTest(
            test_items=[item_1, item_2],
            annotators={"some-annotator": FakeAnnotator()},
            measurement=_FAKE_MEASUREMENT,
        ),
        FakeSUT(),
        tmpdir,
    )

    assert record.test_item_records == [
        _make_test_item_record(item_1),
        _make_test_item_record(item_2),
    ]
    assert record.result.to_instance() == FakeTestResult(count_test_items=2.0)


def test_run_prompt_response_test_caching(tmpdir):
    test_items = [fake_test_item("1")]
    annotator_1 = FakeAnnotator()
    sut_1 = FakeSUT()
    # First run is in empty directory
    record_1 = run_prompt_response_test(
        FakeTest(
            test_items=test_items,
            annotators={"some-annotator": annotator_1},
            measurement=_FAKE_MEASUREMENT,
        ),
        sut_1,
        tmpdir,
    )
    assert sut_1.evaluate_calls == 1
    assert annotator_1.annotate_calls == 1
    # Second run should be fully cached
    annotator_2 = FakeAnnotator()
    sut_2 = FakeSUT()
    record_2 = run_prompt_response_test(
        FakeTest(
            test_items=test_items,
            annotators={"some-annotator": annotator_2},
            measurement=_FAKE_MEASUREMENT,
        ),
        sut_2,
        tmpdir,
    )
    assert sut_2.evaluate_calls == 0
    assert annotator_2.annotate_calls == 0
    # Fields like timestamp and initialization differ, so ignore them.
    assert record_1.test_item_records == record_2.test_item_records
    assert record_1.result == record_2.result


def test_run_prompt_response_test_ignore_caching(tmpdir):
    test_items = [fake_test_item("1")]
    annotator_1 = FakeAnnotator()
    sut_1 = FakeSUT()
    # First run is in empty directory, turn off caching.
    record_1 = run_prompt_response_test(
        FakeTest(
            test_items=test_items,
            annotators={"some-annotator": annotator_1},
            measurement=_FAKE_MEASUREMENT,
        ),
        sut_1,
        tmpdir,
        use_caching=False,
    )
    assert sut_1.evaluate_calls == 1
    assert annotator_1.annotate_calls == 1
    # Second run even with the same objects should call again.
    record_2 = run_prompt_response_test(
        FakeTest(
            test_items=test_items,
            annotators={"some-annotator": annotator_1},
            measurement=_FAKE_MEASUREMENT,
        ),
        sut_1,
        tmpdir,
    )
    assert sut_1.evaluate_calls == 2
    assert annotator_1.annotate_calls == 2
    # Fields like timestamp and initialization differ, so ignore them.
    assert record_1.test_item_records == record_2.test_item_records
    assert record_1.result == record_2.result


def fake_run(max_test_items, tmpdir):
    # Lots of test items
    test_items = [fake_test_item(str(i)) for i in range(100)]
    record = run_prompt_response_test(
        FakeTest(
            test_items=test_items,
            annotators={"some-annotator": FakeAnnotator()},
            measurement=_FAKE_MEASUREMENT,
        ),
        FakeSUT(),
        tmpdir,
        # Limit to just 3 test items
        max_test_items=max_test_items,
    )
    return record


def test_run_prompt_response_test_max_test_items(tmpdir):
    max_test_items = 3
    record = fake_run(max_test_items, tmpdir)
    assert len(record.test_item_records) == max_test_items
    assert record.result.to_instance() == FakeTestResult(count_test_items=3.0)


def test_run_prompt_response_test_max_test_items_stable(tmpdir):
    run3 = fake_run(3, tmpdir)
    run4 = fake_run(4, tmpdir)
    prompts3 = [r.test_item.prompts[0].prompt.text for r in run3.test_item_records]
    prompts4 = [r.test_item.prompts[0].prompt.text for r in run4.test_item_records]
    assert len(prompts3) == 3
    assert len(prompts4) == 4

    for p in prompts3:
        assert p in prompts4


def test_run_prompt_response_test_max_test_items_zero(tmpdir):
    # Lots of test items
    test_items = [fake_test_item(str(i)) for i in range(100)]
    with pytest.raises(AssertionError) as err_info:
        run_prompt_response_test(
            FakeTest(
                test_items=test_items,
                annotators={"some-annotator": FakeAnnotator()},
                measurement={},
            ),
            FakeSUT(),
            tmpdir,
            max_test_items=0,
        )
    assert str(err_info.value) == "Cannot run a test using 0."


@pytest.mark.parametrize(
    "exception_source", ["evaluate", "translate_text_prompt", "translate_response"]
)
def test_run_prompt_response_test_sut_exception(exception_source, tmpdir):
    test_item = fake_test_item("1")
    sut = FakeSUT()

    def raise_exception(*args, **kwargs):
        raise Exception("some-exception")

    setattr(sut, exception_source, raise_exception)

    record = run_prompt_response_test(
        FakeTest(
            test_items=[test_item],
            annotators={"some-annotator": FakeAnnotator()},
            measurement=_FAKE_MEASUREMENT,
        ),
        sut,
        tmpdir,
    )

    assert record.test_item_exceptions == [_make_sut_exception_record(test_item)]


@pytest.mark.parametrize(
    "exception_source", ["annotate", "translate_request", "translate_response"]
)
def test_run_prompt_response_test_annotator_exception(exception_source, tmpdir):
    test_item = fake_test_item("1")
    annotator = FakeAnnotator()

    def raise_exception(*args, **kwargs):
        raise Exception("some-exception")

    setattr(annotator, exception_source, raise_exception)

    record = run_prompt_response_test(
        FakeTest(
            test_items=[test_item],
            annotators={"some-annotator": annotator},
            measurement=_FAKE_MEASUREMENT,
        ),
        FakeSUT(),
        tmpdir,
    )

    assert record.test_item_exceptions == [_make_annotator_exception_record(test_item)]


def unreliable_sut(trigger_test_item):
    sut = FakeSUT()
    original_evaluate = sut.evaluate

    def _side_effect(request):
        if request.text == trigger_test_item.prompts[0].prompt.text:
            raise Exception("some-exception")
        return original_evaluate(request)

    sut.evaluate = mock.Mock(side_effect=_side_effect)
    return sut


def unreliable_annotator(trigger_test_item):
    annotator = FakeAnnotator()
    original_annotate = annotator.annotate

    def _side_effect(request):
        if request.text == trigger_test_item.prompts[0].prompt.text:
            raise Exception("some-exception")
        return original_annotate(request)

    annotator.annotate = mock.Mock(side_effect=_side_effect)
    return annotator


def test_run_prompt_response_test_output_multiple_exceptions(tmpdir):
    item_1 = fake_test_item("1")
    item_2 = fake_test_item("2")
    sut_trigger_item = fake_test_item("bad sut")
    annotator_trigger_item = fake_test_item("bad annotator")

    sut = unreliable_sut(sut_trigger_item)
    annotator = unreliable_annotator(annotator_trigger_item)

    record = run_prompt_response_test(
        FakeTest(
            test_items=[item_1, sut_trigger_item, annotator_trigger_item, item_2],
            annotators={"some-annotator": annotator},
            measurement=_FAKE_MEASUREMENT,
        ),
        sut,
        tmpdir,
    )

    assert record.test_item_records == [
        _make_test_item_record(item_1),
        _make_test_item_record(item_2),
    ]
    assert record.test_item_exceptions == [
        _make_sut_exception_record(sut_trigger_item),
        _make_annotator_exception_record(annotator_trigger_item),
    ]
    assert record.result.to_instance() == FakeTestResult(count_test_items=2.0)


def test_run_prompt_response_test_invalid_result(tmpdir):
    sut_trigger_item = fake_test_item("bad sut")
    sut = unreliable_sut(sut_trigger_item)

    record = run_prompt_response_test(
        FakeTest(
            test_items=[sut_trigger_item],
            annotators={"some-annotator": FakeAnnotator()},
            measurement=_FAKE_MEASUREMENT,
        ),
        sut,
        tmpdir,
    )

    assert len(record.test_item_records) == 0
    assert record.result.to_instance() == {"Unknown test result": "no valid items."}


def test_run_prompt_response_test_good_cache_on_annotator_translate_exception(tmpdir):
    annotator = FakeAnnotator()

    def _raise_exception(*args, **kwargs):
        raise Exception("some-exception")

    annotator.translate_response = _raise_exception

    run_prompt_response_test(
        FakeTest(
            test_items=[(fake_test_item("1"))],
            annotators={"some-annotator": annotator},
            measurement=_FAKE_MEASUREMENT,
        ),
        FakeSUT(),
        tmpdir,
    )

    annotator_cache = SqlDictCache(
        os.path.join(tmpdir, "tests/FakeTest/annotators"), "some-annotator"
    )
    with annotator_cache.cached_responses as cache:
        assert len(cache) == 0


class NotATestOrSut:
    pass


def test_run_prompt_response_test_invalid_test(tmpdir):
    with pytest.raises(AssertionError) as err_info:
        run_prompt_response_test(
            NotATestOrSut(),
            FakeSUT(),
            tmpdir,
        )
    assert (
        str(err_info.value)
        == "NotATestOrSut should be decorated with @modelgauge_test."
    )


def test_run_prompt_response_test_invalid_sut(tmpdir):
    with pytest.raises(AssertionError) as err_info:
        run_prompt_response_test(
            FakeTest(),
            NotATestOrSut(),
            tmpdir,
        )
    assert (
        str(err_info.value) == "NotATestOrSut should be decorated with @modelgauge_sut."
    )


@modelgauge_test(requires_sut_capabilities=[ProducesPerTokenLogProbabilities])
class FakeTestWithReqs(FakeTest):
    pass


def test_run_prompt_response_test_missing_capabilities(tmpdir):
    with pytest.raises(AssertionError) as err_info:
        run_prompt_response_test(
            FakeTestWithReqs(),
            FakeSUT(),
            tmpdir,
        )
    assert "Test test-uid cannot run on fake-sut" in str(err_info.value)
    assert "ProducesPerTokenLogProbabilities" in str(err_info.value)
