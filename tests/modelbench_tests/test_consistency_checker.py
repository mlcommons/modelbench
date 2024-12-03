import json
import pytest
import re
from click.testing import CliRunner, Result
from typing import Dict, List

from modelbench import run
from modelbench.consistency_checker import ConsistencyChecker

DEFAULT_SUT = "sut1"
DEFAULT_TEST = "test1"
DEFAULT_ANNOTATOR = "annotator1"


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


def make_basic_run(suts: List[str], test_prompts: Dict[str, List[str]], annotators: List[str]):
    """Successful "fresh" benchmark run with all SUT/annotator responses fetched (not cached).
    Measurements/annotations are all safe."""

    journal = []
    journal.append({"message": "starting run", "suts": suts, "tests": list(test_prompts.keys())})
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
    return journal


@pytest.fixture
def basic_benchmark_run():
    return make_basic_run(
        suts=["sut1", "sut2"],
        test_prompts={"test1": ["prompt1", "prompt2"]},
        annotators=["annotator1", "annotator2", "annotator3"],
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


def test_cached_and_fetched_only_annotators_also_collected(tmp_path, basic_benchmark_run):
    basic_benchmark_run.append(make_sut_entry("fetched annotator response", annotator="annotator4"))
    basic_benchmark_run.append(make_sut_entry("using cached annotator response", annotator="annotator5"))
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)

    assert "annotator4" in checker.annotators
    assert "annotator5" in checker.annotators


@pytest.mark.parametrize(
    "duplicate_message,failed_check",
    [
        ("queuing item", "EachPromptQueuedOnce"),
        ("fetched sut response", "EachPromptRespondedToOnce"),
        ("using cached sut response", "EachPromptRespondedToOnce"),
        ("translated sut response", "EachResponseTranslatedOnce"),
        ("measured item quality", "EachItemMeasuredOnce"),
    ],
)
def test_run_with_duplicate_sut_stuff(tmp_path, basic_benchmark_run, duplicate_message, failed_check):
    basic_benchmark_run.append(make_sut_entry(duplicate_message))
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)
    checker.run()

    subchecker = checker.test_sut_level_checker
    failed_row = subchecker._row_key(sut=DEFAULT_SUT, test=DEFAULT_TEST)
    assert subchecker.check_is_complete()
    assert subchecker.results[failed_row][failed_check] is False
    # TODO: Check warnings


@pytest.mark.parametrize(
    "extra_earlier_message,failed_check",
    [
        ("queuing item", "EachPromptRespondedToOnce"),
        ("fetched sut response", "EachResponseTranslatedOnce"),
        ("translated sut response", "EachItemMeasuredOnce"),
    ],
)
def test_run_with_missing_sut_stuff(tmp_path, basic_benchmark_run, extra_earlier_message, failed_check):
    basic_benchmark_run.append(make_sut_entry(extra_earlier_message, prompt_id="NEW PROMPT"))
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)
    checker.run()

    subchecker = checker.test_sut_level_checker
    failed_row = subchecker._row_key(sut=DEFAULT_SUT, test=DEFAULT_TEST)
    assert subchecker.check_is_complete()
    assert subchecker.results[failed_row][failed_check] is False
    # TODO: Check warnings


def test_run_with_missing_queued_item_for_sut(tmp_path, basic_benchmark_run):
    # Add extra test item by adding an entry for another sut.
    basic_benchmark_run.append(make_sut_entry("queuing item", sut="another_sut", prompt_id="NEW PROMPT"))
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)
    checker.run()

    subchecker = checker.test_sut_level_checker
    failed_row = subchecker._row_key(sut=DEFAULT_SUT, test=DEFAULT_TEST)
    assert subchecker.check_is_complete()
    assert subchecker.results[failed_row]["EachPromptQueuedOnce"] is False
    # TODO: Check warnings


@pytest.mark.parametrize(
    "extra_message,failed_check",
    [
        ("fetched sut response", "EachPromptRespondedToOnce"),
        ("translated sut response", "EachResponseTranslatedOnce"),
        ("measured item quality", "EachItemMeasuredOnce"),
        ("measured item quality", "NumItemsFinishedEqualsMeasuredItems"),
    ],
)
def test_run_with_extra_sut_stuff(tmp_path, basic_benchmark_run, extra_message, failed_check):
    basic_benchmark_run.append(make_sut_entry(extra_message, prompt_id="NEW PROMPT"))
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)
    checker.run()

    subchecker = checker.test_sut_level_checker
    failed_row = subchecker._row_key(sut=DEFAULT_SUT, test=DEFAULT_TEST)
    assert subchecker.check_is_complete()
    assert subchecker.results[failed_row][failed_check] is False
    # TODO: Check warnings


@pytest.mark.parametrize(
    "duplicate_message,failed_check",
    [
        ("fetched annotator response", "EachResponseAnnotatedOnce"),
        ("using cached annotator response", "EachResponseAnnotatedOnce"),
        ("translated annotation", "EachAnnotationTranslatedOnce"),
    ],
)
def test_run_with_duplicate_annotator_stuff(tmp_path, basic_benchmark_run, duplicate_message, failed_check):
    basic_benchmark_run.append(make_sut_entry(duplicate_message))
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)
    checker.run()

    subchecker = checker.test_sut_annotator_level_checker
    failed_row = subchecker._row_key(sut=DEFAULT_SUT, test=DEFAULT_TEST, annotator=DEFAULT_ANNOTATOR)
    assert subchecker.check_is_complete()
    assert subchecker.results[failed_row][failed_check] is False
    # TODO: Check warnings


def test_run_with_missing_annotations(tmp_path, basic_benchmark_run):
    basic_benchmark_run.append(make_sut_entry("translated sut response", prompt_id="NEW PROMPT"))
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)
    checker.run()

    subchecker = checker.test_sut_annotator_level_checker
    failed_row = subchecker._row_key(sut=DEFAULT_SUT, test=DEFAULT_TEST, annotator=DEFAULT_ANNOTATOR)
    assert subchecker.check_is_complete()
    assert subchecker.results[failed_row]["EachResponseAnnotatedOnce"] is False
    # TODO: Check warnings


@pytest.mark.parametrize(
    "extra_earlier_message,failed_check",
    [
        ("fetched annotator response", "EachAnnotationTranslatedOnce"),
        ("using cached annotator response", "EachAnnotationTranslatedOnce"),
    ],
)
def test_run_with_missing_annotator_translations(tmp_path, basic_benchmark_run, extra_earlier_message, failed_check):
    basic_benchmark_run.append(make_sut_entry(extra_earlier_message, prompt_id="NEW PROMPT"))
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)
    checker.run()

    subchecker = checker.test_sut_annotator_level_checker
    failed_row = subchecker._row_key(sut=DEFAULT_SUT, test=DEFAULT_TEST, annotator=DEFAULT_ANNOTATOR)
    assert subchecker.check_is_complete()
    assert subchecker.results[failed_row][failed_check] is False
    # TODO: Check warnings


@pytest.mark.parametrize(
    "extra_message,failed_check",
    [
        ("fetched annotator response", "EachResponseAnnotatedOnce"),
        ("using cached annotator response", "EachResponseAnnotatedOnce"),
        ("translated annotation", "EachAnnotationTranslatedOnce"),
    ],
)
def test_run_with_extra_annotator_stuff(tmp_path, basic_benchmark_run, extra_message, failed_check):
    basic_benchmark_run.append(make_sut_entry(extra_message, prompt_id="NEW PROMPT"))
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)
    checker.run()

    subchecker = checker.test_sut_annotator_level_checker
    failed_row = subchecker._row_key(sut=DEFAULT_SUT, test=DEFAULT_TEST, annotator=DEFAULT_ANNOTATOR)
    assert subchecker.check_is_complete()
    assert subchecker.results[failed_row][failed_check] is False
    # TODO: Check warnings


# TODO: Add tests for AnnotationsMergedCorrectly checker.


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
    assert subchecker.results[failed_row]["MinValidAnnotatorItems"] is False
    # TODO: Check warnings


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


def run_cli(*args) -> Result:
    # noinspection PyTypeChecker
    result = CliRunner().invoke(run.cli, args, catch_exceptions=False)
    return result


def journal_result_is_expected_in_summary(journal_path, expected_result, output):
    f_result = ConsistencyChecker.format_result(expected_result)
    return re.search(rf"{re.escape(str(journal_path))}\s*.*\s*{f_result}", output)


def test_summarize_multiple_journal_checks(tmp_path, basic_benchmark_run):
    journal1_path = tmp_path / "journal-run-1.jsonl"
    write_journal_to_file(basic_benchmark_run, journal1_path)
    journal2_path = tmp_path / "journal-run-2.jsonl"
    write_journal_to_file(basic_benchmark_run, journal2_path)

    result = run_cli("consistency-check", str(tmp_path))

    assert result.exit_code == 0
    # Check that both journal checks are marked as "passed" in the summary.
    assert journal_result_is_expected_in_summary(journal1_path, True, result.output)
    assert journal_result_is_expected_in_summary(journal2_path, True, result.output)


def test_summarize_multiple_journal_checks_with_fails(tmp_path, basic_benchmark_run):
    journal1_path = tmp_path / "journal-run-1.jsonl"
    write_journal_to_file(basic_benchmark_run, journal1_path)
    # Second journal should fail checks.
    basic_benchmark_run.append(
        {"message": "fetched sut response", "test": "test1", "sut": "sut1", "prompt_id": "NEW PROMPT"}
    )
    journal2_path = tmp_path / "journal-run-2.jsonl"
    write_journal_to_file(basic_benchmark_run, journal2_path)

    result = run_cli("consistency-check", str(tmp_path))

    assert result.exit_code == 0
    assert journal_result_is_expected_in_summary(journal1_path, True, result.output)
    assert journal_result_is_expected_in_summary(journal2_path, False, result.output)
