from newhelm.annotation import Annotation
from newhelm.base_test import Result
from newhelm.records import TestItemRecord
from newhelm.runners.simple_test_runner import run_prompt_response_test
from newhelm.single_turn_prompt_response import (
    PromptInteraction,
)
from newhelm.sut import SUTCompletion, SUTResponse
from tests.fake_annotator import FakeAnnotator
from tests.fake_sut import FakeSUT
from tests.fake_test import FakeTest, fake_test_item


def test_run_prompt_response_test_output(tmpdir):
    item_1 = fake_test_item("1")
    item_2 = fake_test_item("2")
    fake_measurement = {"some-measurement": 0.5}
    record = run_prompt_response_test(
        "some-test",
        FakeTest(
            test_items=[item_1, item_2],
            annotators={"some-annotator": FakeAnnotator()},
            measurement=fake_measurement,
        ),
        "some-sut",
        FakeSUT(),
        tmpdir,
    )

    assert record.test_item_records == [
        TestItemRecord(
            test_item=item_1,
            interactions=[
                PromptInteraction(
                    prompt=item_1.prompts[0],
                    response=SUTResponse(completions=[SUTCompletion(text="1")]),
                )
            ],
            annotations={
                "some-annotator": Annotation(
                    module="tests.fake_annotator",
                    class_name="FakeAnnotation",
                    data={"sut_text": "1"},
                )
            },
            measurements=fake_measurement,
        ),
        TestItemRecord(
            test_item=item_2,
            interactions=[
                PromptInteraction(
                    prompt=item_2.prompts[0],
                    response=SUTResponse(completions=[SUTCompletion(text="2")]),
                )
            ],
            annotations={
                "some-annotator": Annotation(
                    module="tests.fake_annotator",
                    class_name="FakeAnnotation",
                    data={"sut_text": "2"},
                )
            },
            measurements=fake_measurement,
        ),
    ]
    assert record.results == [Result(name="count_test_items", value=2.0)]


def test_run_prompt_response_test_caching(tmpdir):
    test_items = [fake_test_item("1")]
    fake_measurement = {"some-measurement": 0.5}
    annotator_1 = FakeAnnotator()
    sut_1 = FakeSUT()
    # First run is in empty directory
    record_1 = run_prompt_response_test(
        "some-test",
        FakeTest(
            test_items=test_items,
            annotators={"some-annotator": annotator_1},
            measurement=fake_measurement,
        ),
        "some-sut",
        sut_1,
        tmpdir,
    )
    assert sut_1.evaluate_calls == 1
    assert annotator_1.annotate_test_item_calls == 1
    # Second run should be fully cached
    annotator_2 = FakeAnnotator()
    sut_2 = FakeSUT()
    record_2 = run_prompt_response_test(
        "some-test",
        FakeTest(
            test_items=test_items,
            annotators={"some-annotator": annotator_2},
            measurement=fake_measurement,
        ),
        "some-sut",
        sut_2,
        tmpdir,
    )
    assert sut_2.evaluate_calls == 0
    assert annotator_2.annotate_test_item_calls == 0
    # Fields like timestamp and initialization differ, so ignore them.
    assert record_1.test_item_records == record_2.test_item_records
    assert record_1.results == record_2.results


def test_run_prompt_response_test_ignore_caching(tmpdir):
    test_items = [fake_test_item("1")]
    fake_measurement = {"some-measurement": 0.5}
    annotator_1 = FakeAnnotator()
    sut_1 = FakeSUT()
    # First run is in empty directory, turn off caching.
    record_1 = run_prompt_response_test(
        "some-test",
        FakeTest(
            test_items=test_items,
            annotators={"some-annotator": annotator_1},
            measurement=fake_measurement,
        ),
        "some-sut",
        sut_1,
        tmpdir,
        use_caching=False,
    )
    assert sut_1.evaluate_calls == 1
    assert annotator_1.annotate_test_item_calls == 1
    # Second run even with the same objects should call again.
    record_2 = run_prompt_response_test(
        "some-test",
        FakeTest(
            test_items=test_items,
            annotators={"some-annotator": annotator_1},
            measurement=fake_measurement,
        ),
        "some-sut",
        sut_1,
        tmpdir,
    )
    assert sut_1.evaluate_calls == 2
    assert annotator_1.annotate_test_item_calls == 2
    # Fields like timestamp and initialization differ, so ignore them.
    assert record_1.test_item_records == record_2.test_item_records
    assert record_1.results == record_2.results


def test_run_prompt_response_test_max_test_items(tmpdir):
    # Lots of test items
    test_items = [fake_test_item(str(i)) for i in range(100)]
    fake_measurement = {"some-measurement": 0.5}
    record = run_prompt_response_test(
        "some-test",
        FakeTest(
            test_items=test_items,
            annotators={"some-annotator": FakeAnnotator()},
            measurement=fake_measurement,
        ),
        "some-sut",
        FakeSUT(),
        tmpdir,
        # Limit to just 3 test items
        max_test_items=3,
    )
    assert len(record.test_item_records) == 3
    assert record.results == [Result(name="count_test_items", value=3.0)]
