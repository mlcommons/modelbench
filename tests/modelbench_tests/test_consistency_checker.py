import json
import pytest
import re
from typing import Dict, List

from modelbench.consistency_checker import (
    AnnotationsMergedCorrectly,
    ConsistencyChecker,
    EachAnnotationTranslatedOnce,
    EachItemMeasuredOnce,
    EachPromptQueuedOnce,
    EachPromptRespondedToOnce,
    EachResponseAnnotatedOnce,
    EachResponseTranslatedOnce,
    HazardScoreIsFracSafe,
    MinValidAnnotatorItems,
    NumItemsFinishedEqualsMeasuredItems,
    summarize_consistency_check_results,
)
from modelbench.scoring import score_to_ordinal_grade

DEFAULT_SUT = "sut1"
DEFAULT_TEST = "test1"
DEFAULT_ANNOTATOR = "annotator1"
DEFAULT_HAZARD = "hazard1"


def make_sut_entry(
    message,
    test=DEFAULT_TEST,
    sut=DEFAULT_SUT,
    prompt_id="prompt1",
    annotator=DEFAULT_ANNOTATOR,
    measurements_is_safe=1.0,
    translated_is_safe=True,
):
    """Helper function to create a basic SUT journal entry. Not all args will be used depending on the messge."""
    # SUT messages
    base_sut_entry = {"test": test, "sut": sut, "prompt_id": prompt_id}
    if message in ["queuing item", "fetched sut response", "using cached sut response", "translated sut response"]:
        # Basic SUT journal entries that share the same shape.
        return {"message": message, **base_sut_entry}
    if message == "measured item quality":
        return {"message": message, "measurements": {"is_safe": measurements_is_safe}, **base_sut_entry}

    # SUT + Annotator messages
    base_annotator_entry = {"annotator": annotator, **base_sut_entry}
    if message in ["fetched annotator response", "using cached annotator response"]:
        # Basic annotator journal entries that share the same shape.
        return {"message": message, **base_annotator_entry}
    if message == "translated annotation":
        return {"message": message, "annotation": {"is_safe": translated_is_safe}, **base_annotator_entry}

    raise ValueError(f"Can't make SUT entry for message: {message}")


def make_basic_run(
    suts: List[str], test_prompts: Dict[str, List[str]], annotators: List[str], hazard_tests: Dict[str, List[str]]
):
    """Successful "fresh" benchmark run with all SUT/annotator responses fetched (not cached).
    Measurements/annotations are all safe.
    Each hazard uses all tests."""
    benchmark = "official"
    journal = []
    journal.append(
        {"message": "starting run", "suts": suts, "tests": list(test_prompts.keys()), "benchmarks": [benchmark]}
    )
    for hazard, tests in hazard_tests.items():
        journal.append({"message": "hazard info", "hazard": hazard, "benchmark": benchmark, "tests": tests})
    for sut in suts:
        for test, prompts in test_prompts.items():
            journal.append({"message": "using test items", "test": test, "using": len(prompts)})
            for prompt in prompts:
                # Normal SUT pipeline.
                base_sut_entry = {"test": test, "sut": sut, "prompt_id": prompt}
                for message in [
                    "queuing item",
                    "fetched sut response",
                    "translated sut response",
                    "measured item quality",
                ]:
                    journal.append(make_sut_entry(message, **base_sut_entry))
                # Annotator pipeline.
                for annotator in annotators:
                    for message in ["fetched annotator response", "translated annotation"]:
                        journal.append(make_sut_entry(message, annotator=annotator, **base_sut_entry))
            journal.append({"message": "test scored", "test": test, "sut": sut, "items_finished": len(prompts)})
        for hazard, tests in hazard_tests.items():
            journal.append(
                {
                    "message": "hazard scored",
                    "benchmark": benchmark,
                    "hazard": hazard,
                    "sut": sut,
                    "score": 1.0,
                    "reference": 0.9,
                    "samples": sum(len(test_prompts[test]) for test in tests),
                    "numeric_grade": score_to_ordinal_grade(1.0, 0.9),
                }
            )
    return journal


@pytest.fixture
def basic_benchmark_run():
    return make_basic_run(
        suts=["sut1", "sut2"],
        test_prompts={"test1": ["prompt1", "prompt2"]},
        annotators=["annotator1", "annotator2", "annotator3"],
        hazard_tests={"hazard1": ["test1"]},
    )


def write_journal_to_file(journal, path):
    with open(path, "w") as f:
        for item in journal:
            f.write(json.dumps(item) + "\n")


def init_checker_for_journal(tmp_path, journal):
    journal_path = tmp_path / "journal.jsonl"
    write_journal_to_file(journal, journal_path)
    checker = ConsistencyChecker(journal_path=journal_path)
    return checker


def test_normal_run(tmp_path, basic_benchmark_run):
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)
    checker.run()

    for subchecker in [checker.test_sut_level_checker, checker.test_sut_annotator_level_checker]:
        assert subchecker.check_is_complete()
        for row in subchecker.results.values():
            assert all(row)
        assert subchecker.warnings == []


def test_entities_collected(tmp_path, basic_benchmark_run):
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)

    assert sorted(checker.suts) == ["sut1", "sut2"]
    assert checker.tests == ["test1"]
    assert sorted(checker.annotators) == ["annotator1", "annotator2", "annotator3"]
    assert checker.hazards == ["hazard1"]


def test_cached_and_fetched_only_annotators_also_collected(tmp_path, basic_benchmark_run):
    basic_benchmark_run.append(make_sut_entry("fetched annotator response", annotator="annotator4"))
    basic_benchmark_run.append(make_sut_entry("using cached annotator response", annotator="annotator5"))
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)

    assert "annotator4" in checker.annotators
    assert "annotator5" in checker.annotators


@pytest.mark.parametrize(
    "duplicate_message,failed_check",
    [
        ("queuing item", EachPromptQueuedOnce),
        ("fetched sut response", EachPromptRespondedToOnce),
        ("using cached sut response", EachPromptRespondedToOnce),
        ("translated sut response", EachResponseTranslatedOnce),
        ("measured item quality", EachItemMeasuredOnce),
    ],
)
def test_run_with_duplicate_sut_stuff(tmp_path, basic_benchmark_run, duplicate_message, failed_check):
    basic_benchmark_run.append(make_sut_entry(duplicate_message))
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)
    checker.run()

    subchecker = checker.test_sut_level_checker
    failed_row = subchecker._row_key(sut=DEFAULT_SUT, test=DEFAULT_TEST)
    assert subchecker.check_is_complete()
    assert subchecker.results[failed_row][subchecker._col_name(failed_check)] is False
    # TODO: Check warnings


@pytest.mark.parametrize(
    "extra_earlier_message,failed_check",
    [
        ("queuing item", EachPromptRespondedToOnce),
        ("fetched sut response", EachResponseTranslatedOnce),
        ("translated sut response", EachItemMeasuredOnce),
    ],
)
def test_run_with_missing_sut_stuff(tmp_path, basic_benchmark_run, extra_earlier_message, failed_check):
    basic_benchmark_run.append(make_sut_entry(extra_earlier_message, prompt_id="NEW PROMPT"))
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)
    checker.run()

    subchecker = checker.test_sut_level_checker
    failed_row = subchecker._row_key(sut=DEFAULT_SUT, test=DEFAULT_TEST)
    assert subchecker.check_is_complete()
    assert subchecker.results[failed_row][subchecker._col_name(failed_check)] is False
    # TODO: Check warnings


def test_run_with_missing_queued_item_for_sut(tmp_path, basic_benchmark_run):
    # Add extra test item by adding an entry for another sut.
    basic_benchmark_run.append(make_sut_entry("queuing item", sut="another_sut", prompt_id="NEW PROMPT"))
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)
    checker.run()

    subchecker = checker.test_sut_level_checker
    failed_row = subchecker._row_key(sut=DEFAULT_SUT, test=DEFAULT_TEST)
    assert subchecker.check_is_complete()
    assert subchecker.results[failed_row][subchecker._col_name(EachPromptQueuedOnce)] is False
    # TODO: Check warnings


@pytest.mark.parametrize(
    "extra_message,failed_check",
    [
        ("fetched sut response", EachPromptRespondedToOnce),
        ("translated sut response", EachResponseTranslatedOnce),
        ("measured item quality", EachItemMeasuredOnce),
        ("measured item quality", NumItemsFinishedEqualsMeasuredItems),
    ],
)
def test_run_with_extra_sut_stuff(tmp_path, basic_benchmark_run, extra_message, failed_check):
    basic_benchmark_run.append(make_sut_entry(extra_message, prompt_id="NEW PROMPT"))
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)
    checker.run()

    subchecker = checker.test_sut_level_checker
    failed_row = subchecker._row_key(sut=DEFAULT_SUT, test=DEFAULT_TEST)
    assert subchecker.check_is_complete()
    assert subchecker.results[failed_row][subchecker._col_name(failed_check)] is False
    # TODO: Check warnings


@pytest.mark.parametrize(
    "duplicate_message,failed_check",
    [
        ("fetched annotator response", EachResponseAnnotatedOnce),
        ("using cached annotator response", EachResponseAnnotatedOnce),
        ("translated annotation", EachAnnotationTranslatedOnce),
    ],
)
def test_run_with_duplicate_annotator_stuff(tmp_path, basic_benchmark_run, duplicate_message, failed_check):
    basic_benchmark_run.append(make_sut_entry(duplicate_message))
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)
    checker.run()

    subchecker = checker.test_sut_annotator_level_checker
    failed_row = subchecker._row_key(sut=DEFAULT_SUT, test=DEFAULT_TEST, annotator=DEFAULT_ANNOTATOR)
    assert subchecker.check_is_complete()
    assert subchecker.results[failed_row][subchecker._col_name(failed_check)] is False
    # TODO: Check warnings


def test_run_with_missing_annotations(tmp_path, basic_benchmark_run):
    basic_benchmark_run.append(make_sut_entry("translated sut response", prompt_id="NEW PROMPT"))
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)
    checker.run()

    subchecker = checker.test_sut_annotator_level_checker
    failed_row = subchecker._row_key(sut=DEFAULT_SUT, test=DEFAULT_TEST, annotator=DEFAULT_ANNOTATOR)
    assert subchecker.check_is_complete()
    assert subchecker.results[failed_row][subchecker._col_name(EachResponseAnnotatedOnce)] is False
    # TODO: Check warnings


@pytest.mark.parametrize(
    "extra_earlier_message,failed_check",
    [
        ("fetched annotator response", EachAnnotationTranslatedOnce),
        ("using cached annotator response", EachAnnotationTranslatedOnce),
    ],
)
def test_run_with_missing_annotator_translations(tmp_path, basic_benchmark_run, extra_earlier_message, failed_check):
    basic_benchmark_run.append(make_sut_entry(extra_earlier_message, prompt_id="NEW PROMPT"))
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)
    checker.run()

    subchecker = checker.test_sut_annotator_level_checker
    failed_row = subchecker._row_key(sut=DEFAULT_SUT, test=DEFAULT_TEST, annotator=DEFAULT_ANNOTATOR)
    assert subchecker.check_is_complete()
    assert subchecker.results[failed_row][subchecker._col_name(failed_check)] is False
    # TODO: Check warnings


@pytest.mark.parametrize(
    "extra_message,failed_check",
    [
        ("fetched annotator response", EachResponseAnnotatedOnce),
        ("using cached annotator response", EachResponseAnnotatedOnce),
        ("translated annotation", EachAnnotationTranslatedOnce),
    ],
)
def test_run_with_extra_annotator_stuff(tmp_path, basic_benchmark_run, extra_message, failed_check):
    basic_benchmark_run.append(make_sut_entry(extra_message, prompt_id="NEW PROMPT"))
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)
    checker.run()

    subchecker = checker.test_sut_annotator_level_checker
    failed_row = subchecker._row_key(sut=DEFAULT_SUT, test=DEFAULT_TEST, annotator=DEFAULT_ANNOTATOR)
    assert subchecker.check_is_complete()
    assert subchecker.results[failed_row][subchecker._col_name(failed_check)] is False
    # TODO: Check warnings


@pytest.mark.parametrize("is_safe", [True, False])
def test_min_valid_items_checker(tmp_path, basic_benchmark_run, is_safe):
    # Add some invalid translated annotations for one annotator.
    entry = make_sut_entry("translated annotation", prompt_id="NEW PROMPT", translated_is_safe=is_safe)
    entry["annotation"]["is_valid"] = False
    basic_benchmark_run.append(entry)
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)
    checker.run()

    subchecker = checker.test_sut_annotator_level_checker
    failed_row = subchecker._row_key(sut=DEFAULT_SUT, test=DEFAULT_TEST, annotator=DEFAULT_ANNOTATOR)
    assert subchecker.check_is_complete()
    assert subchecker.results[failed_row][subchecker._col_name(MinValidAnnotatorItems)] is False
    # TODO: Check warnings


def test_annotations_merged_correctly(tmp_path, basic_benchmark_run):
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)
    checker.run()

    subchecker = checker.test_sut_level_checker
    failed_row = subchecker._row_key(sut=DEFAULT_SUT, test=DEFAULT_TEST)
    assert subchecker.check_is_complete()
    assert subchecker.results[failed_row][subchecker._col_name(AnnotationsMergedCorrectly)] is True


def test_annotations_merged_correctly_fails_with_one_annotator(tmp_path):
    # Simulate run with only 1 annotator.
    run = make_basic_run(
        suts=["sut1"],
        test_prompts={"test1": ["prompt1"]},
        annotators=["annotator1"],
        hazard_tests={"hazard1": ["test1"]},
    )
    checker = init_checker_for_journal(tmp_path, run)
    checker.run()

    subchecker = checker.test_sut_level_checker
    failed_row = subchecker._row_key(sut="sut1", test="test1")
    assert subchecker.check_is_complete()
    assert subchecker.results[failed_row][subchecker._col_name(AnnotationsMergedCorrectly)] is False


def test_annotations_merged_correctly_false_safe(tmp_path, basic_benchmark_run):
    # Add a bunch of fake unsafe annotations for existing prompt that was measured safe.
    entry = make_sut_entry("translated annotation", translated_is_safe=False)
    for _ in range(4):
        basic_benchmark_run.append(entry)
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)
    checker.run()

    subchecker = checker.test_sut_level_checker
    failed_row = subchecker._row_key(sut=DEFAULT_SUT, test=DEFAULT_TEST)
    assert subchecker.check_is_complete()
    assert subchecker.results[failed_row][subchecker._col_name(AnnotationsMergedCorrectly)] is False


def test_annotations_merged_correctly_false_unsafe(tmp_path, basic_benchmark_run):
    # Create safe annotations for new prompt.
    entry = make_sut_entry("translated annotation", prompt_id="NEW PROMPT", translated_is_safe=True)
    for _ in range(4):
        basic_benchmark_run.append(entry)
    # Measure that prompt as unsafe (wrongly).
    basic_benchmark_run.append(
        make_sut_entry("measured item quality", prompt_id="NEW PROMPT", measurements_is_safe=0.0)
    )
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)
    checker.run()

    subchecker = checker.test_sut_level_checker
    failed_row = subchecker._row_key(sut=DEFAULT_SUT, test=DEFAULT_TEST)
    assert subchecker.check_is_complete()
    assert subchecker.results[failed_row][subchecker._col_name(AnnotationsMergedCorrectly)] is False


def test_hazard_score_fails_with_different_frac_safe(tmp_path, basic_benchmark_run):
    # Add an item that is measured as unsafe and is not counted in the hazard score.
    basic_benchmark_run.append(
        make_sut_entry("measured item quality", measurements_is_safe=0.0, test=DEFAULT_TEST, sut=DEFAULT_SUT)
    )
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)
    checker.run()

    subchecker = checker.hazard_sut_level_checker
    failed_row = subchecker._row_key(hazard=DEFAULT_HAZARD, sut=DEFAULT_SUT)
    assert subchecker.check_is_complete()
    assert subchecker.results[failed_row][subchecker._col_name(HazardScoreIsFracSafe)] is False


def test_hazard_score_skips_with_no_hazard_info_entry(tmp_path):
    """Make sure that the checker still works on older journals that don't provider hazard info."""
    # Make a run without any hazard info entries.
    run = make_basic_run(
        suts=["sut1", "sut2"],
        test_prompts={"test1": ["prompt1", "prompt2"]},
        annotators=["annotator1", "annotator2", "annotator3"],
        hazard_tests={},
    )
    checker = init_checker_for_journal(tmp_path, run)
    assert checker.hazards is None

    checker.run()
    subchecker = checker.hazard_sut_level_checker
    assert subchecker is None


def _manually_set_results_to_pass(sub_checker):
    for row_key in sub_checker.results:
        for col_key in sub_checker.check_names:
            sub_checker.results[row_key][col_key] = True


def test_empty_run_is_not_complete(tmp_path, basic_benchmark_run):
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)
    assert checker.checks_are_complete() is False


def test_partial_run_is_not_complete(tmp_path, basic_benchmark_run):
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)
    # Manually set results for only one sub-checker.
    _manually_set_results_to_pass(checker.test_sut_level_checker)

    assert checker.checks_are_complete() is False


def test_finished_run_is_complete(tmp_path, basic_benchmark_run):
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)
    # Manually set results for all sub-checkers.
    for sub_checker in checker._check_groups:
        _manually_set_results_to_pass(sub_checker)

    assert checker.checks_are_complete()


def journal_result_is_expected_in_summary(journal_path, expected_result, output):
    f_result = ConsistencyChecker.format_result(expected_result)
    return re.search(rf"{re.escape(str(journal_path))}\s*.*\s*{f_result}", output)


def test_summarize_results_pass(tmp_path, capsys, basic_benchmark_run):
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)
    # Manually set results for all sub-checkers.
    for sub_checker in checker._check_groups:
        _manually_set_results_to_pass(sub_checker)
    summarize_consistency_check_results([checker])

    captured = capsys.readouterr()
    assert "✅" in captured.out
    assert "❌" not in captured.out


def test_summarize_results_fail(tmp_path, capsys, basic_benchmark_run):
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)
    for sub_checker in checker._check_groups:
        _manually_set_results_to_pass(sub_checker)
    # Make sure there is at least on failed check.
    checker.test_sut_level_checker.results[("sut1", "test1")][EachPromptQueuedOnce] = False

    summarize_consistency_check_results([checker])

    captured = capsys.readouterr()
    assert "✅" not in captured.out
    assert "❌" in captured.out
