import json
import pytest
import re
from click.testing import CliRunner, Result
from typing import Dict, List

from modelbench import run
from modelbench.consistency_checker import ConsistencyChecker, summarize_consistency_check_results


def make_basic_run(suts: List[str], test_prompts: Dict[str, List[str]], annotators: List[str]):
    """Successful "fresh" benchmark run with all SUT/annotator responses fetched (not cached)."""
    journal = []
    journal.append({"message": "starting run", "suts": suts, "tests": list(test_prompts.keys())})
    for sut in suts:
        for test, prompts in test_prompts.items():
            journal.append({"message": "using test items", "test": test, "using": len(prompts)})
            for prompt in prompts:
                # Normal pipeline.
                sut_messages = [
                    "queuing item",
                    "fetched sut response",
                    "translated sut response",
                    "measured item quality",
                ]
                for message in sut_messages:
                    journal.append({"message": message, "test": test, "prompt_id": prompt, "sut": sut})
                for annotator in annotators:
                    journal.append(
                        {
                            "message": "fetched annotator response",
                            "test": test,
                            "sut": sut,
                            "prompt_id": prompt,
                            "annotator": annotator,
                        }
                    )
    return journal


@pytest.fixture
def basic_benchmark_run():
    return make_basic_run(
        suts=["sut1", "sut2"], test_prompts={"test1": ["prompt1", "prompt2"]}, annotators=["annotator1", "annotator2"]
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


@pytest.mark.parametrize(
    "duplicate_message,failed_check",
    [
        ("fetched sut response", "EachPromptRespondedToOnce"),
        ("using cached sut response", "EachPromptRespondedToOnce"),
        ("translated sut response", "EachResponseTranslatedOnce"),
        ("measured item quality", "EachItemMeasuredOnce"),
    ],
)
def test_run_with_duplicate_sut_stuff(tmp_path, basic_benchmark_run, duplicate_message, failed_check):
    basic_benchmark_run.append({"message": duplicate_message, "test": "test1", "sut": "sut1", "prompt_id": "prompt1"})
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)
    checker.run()

    subchecker = checker.test_sut_level_checker
    failed_row = subchecker._row_key(sut="sut1", test="test1")
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
    basic_benchmark_run.append(
        {"message": extra_earlier_message, "test": "test1", "sut": "sut1", "prompt_id": "NEW PROMPT"}
    )
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)
    checker.run()

    subchecker = checker.test_sut_level_checker
    failed_row = subchecker._row_key(sut="sut1", test="test1")
    assert subchecker.check_is_complete()
    assert subchecker.results[failed_row][failed_check] is False
    # TODO: Check warnings


@pytest.mark.parametrize(
    "extra_message,failed_check",
    [
        ("fetched sut response", "EachPromptRespondedToOnce"),
        ("translated sut response", "EachResponseTranslatedOnce"),
        ("measured item quality", "EachItemMeasuredOnce"),
    ],
)
def test_run_with_extra_sut_stuff(tmp_path, basic_benchmark_run, extra_message, failed_check):
    basic_benchmark_run.append({"message": extra_message, "test": "test1", "sut": "sut1", "prompt_id": "NEW PROMPT"})
    checker = init_checker_for_journal(tmp_path, basic_benchmark_run)
    checker.run()

    subchecker = checker.test_sut_level_checker
    failed_row = subchecker._row_key(sut="sut1", test="test1")
    assert subchecker.check_is_complete()
    assert subchecker.results[failed_row][failed_check] is False
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


def test_summarize_multiple_journal_checks(tmp_path, basic_benchmark_run):
    journal1_path = tmp_path / "journal-run-1.jsonl"
    write_journal_to_file(basic_benchmark_run, journal1_path)
    journal2_path = tmp_path / "journal-run-2.jsonl"
    write_journal_to_file(basic_benchmark_run, journal2_path)

    result = run_cli("consistency-check", "-j", str(tmp_path))

    assert result.exit_code == 0
    # Check that both journal checks are marked as "passed" in the summary.
    PASS = ConsistencyChecker.format_result(True)
    assert re.search(rf"{journal1_path}\s+{PASS}", result.output)
    assert re.search(rf"{journal2_path}\s+{PASS}", result.output)


def test_summarize_multiple_journal_checks_with_fails(tmp_path, basic_benchmark_run):
    journal1_path = tmp_path / "journal-run-1.jsonl"
    write_journal_to_file(basic_benchmark_run, journal1_path)
    # Second journal should fail checks.
    basic_benchmark_run.append({"message": "fetched sut response", "test": "test1", "sut": "sut1", "prompt_id": "NEW PROMPT"})
    journal2_path = tmp_path / "journal-run-2.jsonl"
    write_journal_to_file(basic_benchmark_run, journal2_path)

    result = run_cli("consistency-check", "-j", str(tmp_path))

    assert result.exit_code == 0
    # Check that both journal checks are marked as "passed" in the summary.
    PASS = ConsistencyChecker.format_result(True)
    FAIL = ConsistencyChecker.format_result(False)
    assert re.search(rf"{journal1_path}\s+{PASS}", result.output)
    assert re.search(rf"{journal2_path}\s+{FAIL}", result.output)
