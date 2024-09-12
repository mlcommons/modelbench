import pytest
from modelgauge.prompt import TextPrompt
from modelgauge.single_turn_prompt_response import (
    MeasuredTestItem,
    PromptWithContext,
    TestItem,
)
from modelgauge.suts.openai_client import OpenAIApiKey, OpenAIOrgId
from modelgauge.tests.xstest import (
    StrMatchAnnotator,
    XSTest,
    XSTestItemContext,
    XSTestResult,
)
from tests.fake_dependency_helper import FakeDependencyHelper, make_csv


def _fake_dependency_helper(tmpdir, rows):
    csv_header = ["id_v2", "category", "type", "prompt"]
    csv_text = make_csv(csv_header, rows)
    return FakeDependencyHelper(tmpdir, {"prompts_csv": csv_text})


def _make_test_item(text, is_safe, prompt_id="", prompt_type="") -> TestItem:
    return TestItem(
        prompts=[PromptWithContext(prompt=TextPrompt(text=text), source_id=prompt_id)],
        context=XSTestItemContext(is_safe=is_safe, prompt_type=prompt_type),
    )


def _make_measured_test_item(
    is_safe, openai_label="Full Compliance", strmatch_label="Full Compliance"
) -> MeasuredTestItem:
    """Creates MeasuredTestItem specified openai_annotator and string_match compliance labels."""
    test_item = _make_test_item("", is_safe)
    measurements = {
        "Full Compliance - openai_annotator": 0.0,
        "Full Refusal - openai_annotator": 0.0,
        "Partial Refusal - openai_annotator": 0.0,
        "Full Compliance - string_match": 0.0,
        "Full Refusal - string_match": 0.0,
    }
    measurements[f"{openai_label} - openai_annotator"] = 1.0
    measurements[f"{strmatch_label} - string_match"] = 1.0
    return MeasuredTestItem(test_item=test_item, measurements=measurements)


# I have no idea why these need mypy ignore, but they do.
_FAKE_API_KEY = OpenAIApiKey("some-key")  # type: ignore
_FAKE_ORG_ID = OpenAIOrgId(None)  # type: ignore


def test_make_test_items(tmpdir):
    text_prompts = [
        [
            "id01",
            "Safe - models should comply with / answer the request",
            "homonyms",
            "prompt 1",
        ],
        [
            "id02",
            "Unsafe - models should refuse to comply with / answer",
            "contrast_homonyms",
            "prompt 2",
        ],
    ]
    dependency_helper = _fake_dependency_helper(tmpdir, text_prompts)
    test = XSTest("uid", _FAKE_API_KEY, _FAKE_ORG_ID)
    test_items = test.make_test_items(dependency_helper)

    assert len(test_items) == 2
    assert all([len(item.prompts) == 1 for item in test_items])
    assert all([isinstance(item.context, XSTestItemContext) for item in test_items])
    expected_item_1 = _make_test_item(
        "prompt 1", is_safe=True, prompt_type="homonyms", prompt_id="id01"
    )
    expected_item_2 = _make_test_item(
        "prompt 2", is_safe=False, prompt_type="contrast_homonyms", prompt_id="id02"
    )
    # Ignore the SUTOptions attribute in the prompts
    get_prompts = lambda item: list(map(lambda x: x.prompt.text, item.prompts))
    assert get_prompts(test_items[0]) == get_prompts(expected_item_1)
    assert get_prompts(test_items[1]) == get_prompts(expected_item_2)
    assert test_items[0].context == expected_item_1.context
    assert test_items[1].context == expected_item_2.context


def test_aggregate_openai_measurements():
    measured_test_items = [
        _make_measured_test_item(True, openai_label="Full Compliance"),
        _make_measured_test_item(True, openai_label="Full Refusal"),
        _make_measured_test_item(False, openai_label="Partial Refusal"),
        _make_measured_test_item(False, openai_label="Partial Refusal"),
    ]
    test = XSTest("uid", _FAKE_API_KEY, _FAKE_ORG_ID)
    results = test.aggregate_measurements(measured_test_items)
    expected_results = XSTestResult.model_validate(
        {
            "safe_subset": {
                "openai": {
                    "full_refusal_rate": 0.5,
                    "partial_refusal_rate": 0.0,
                },
                "str_matching": {"full_refusal_rate": 0.0, "partial_refusal_rate": 0.0},
                "subset_size": 2,
            },
            "unsafe_subset": {
                "openai": {"full_refusal_rate": 0.0, "partial_refusal_rate": 1.0},
                "str_matching": {"full_refusal_rate": 0.0, "partial_refusal_rate": 0.0},
                "subset_size": 2,
            },
        }
    )
    assert results == expected_results


def test_aggregate_strmatch_measurements():
    measured_test_items = [
        _make_measured_test_item(True, strmatch_label="Full Compliance"),
        _make_measured_test_item(True, strmatch_label="Full Refusal"),
        _make_measured_test_item(False, strmatch_label="Full Refusal"),
        _make_measured_test_item(False, strmatch_label="Full Refusal"),
    ]
    test = XSTest("uid", _FAKE_API_KEY, _FAKE_ORG_ID)
    results = test.aggregate_measurements(measured_test_items)
    expected_results = XSTestResult.model_validate(
        {
            "safe_subset": {
                "openai": {
                    "full_refusal_rate": 0.0,
                    "partial_refusal_rate": 0.0,
                },
                "str_matching": {"full_refusal_rate": 0.5, "partial_refusal_rate": 0.0},
                "subset_size": 2,
            },
            "unsafe_subset": {
                "openai": {"full_refusal_rate": 0.0, "partial_refusal_rate": 0.0},
                "str_matching": {"full_refusal_rate": 1.0, "partial_refusal_rate": 0.0},
                "subset_size": 2,
            },
        }
    )
    assert results == expected_results


def test_no_contrast_set():
    """Check that no "unsafe" results are returned when there are no unsafe test items."""
    measured_test_items = [
        _make_measured_test_item(True),
        _make_measured_test_item(True),
    ]
    test = XSTest("uid", _FAKE_API_KEY, _FAKE_ORG_ID)
    results = test.aggregate_measurements(measured_test_items)
    assert results.safe_subset.subset_size == 2
    assert results.unsafe_subset is None


def test_invalid_measurements():
    """Tests that assertion is raised for measurements without exactly one true label"""
    test = XSTest("uid", _FAKE_API_KEY, _FAKE_ORG_ID)
    test_item = _make_test_item("prompt", True)
    # Two true openai labels
    measured_test_item_conflicting_labels = MeasuredTestItem(
        test_item=test_item,
        measurements={
            "Full Compliance - openai_annotator": 0.0,
            "Full Refusal - openai_annotator": 1.0,
            "Partial Refusal - openai_annotator": 1.0,
            "Full Compliance - string_match": 0.0,
            "Full Refusal - string_match": 1.0,
        },
    )
    with pytest.raises(Exception, match="Invalid openai_annotator measurements"):
        test.aggregate_measurements([measured_test_item_conflicting_labels])

    # Zero string-match labels
    measured_test_item_zero_strmatch_labels = MeasuredTestItem(
        test_item=test_item,
        measurements={
            "Full Compliance - openai_annotator": 1.0,
            "Full Refusal - openai_annotator": 0.0,
            "Partial Refusal - openai_annotator": 0.0,
            "Full Compliance - string_match": 0.0,
            "Full Refusal - string_match": 0.0,
        },
    )
    with pytest.raises(Exception, match="Invalid string_match measurements"):
        test.aggregate_measurements([measured_test_item_zero_strmatch_labels])


def test_string_match_annotator():
    annotator = StrMatchAnnotator()
    annotated = annotator.annotate({"text": "This is a compliant response."})
    assert annotator.translate_response({}, annotated).compliance == "Full Compliance"
    annotated = annotator.annotate({"text": "I cannot help with this."})
    assert annotator.translate_response({}, annotated).compliance == "Full Refusal"
