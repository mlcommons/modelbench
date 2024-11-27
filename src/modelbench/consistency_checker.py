import json
import shutil
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from itertools import product
from tabulate import tabulate
from typing import Dict, List

from modelbench.run_journal import journal_reader
from modelgauge.config import load_secrets_from_config
from modelgauge.test_registry import TESTS

LINE_WIDTH = shutil.get_terminal_size(fallback=(120, 50)).columns


class JournalSearch:
    def __init__(self, journal_path):
        self.journal_path = journal_path
        self.message_entries: Dict[str, List] = defaultdict(list)  # or maybe sqllite dict?
        # Load journal into message_entries dict.
        self._read_journal()

    def _read_journal(self):
        # Might want to filter out irrelevant messages here. idk.
        with journal_reader(self.journal_path) as f:
            for line in f:
                entry = json.loads(line)
                self.message_entries[entry["message"]].append(entry)

    def query(self, message: str, **kwargs):
        messages = self.message_entries[message]
        return [m for m in messages if all(m[k] == v for k, v in kwargs.items())]

    def num_test_prompts(self, test) -> int:
        # TODO: Implement cache.
        test_entry = self.query("using test items", test=test)
        assert len(test_entry) == 1, "Only 1 `using test items` entry expected per test but found multiple."
        return test_entry[0]["using"]

    def test_prompt_uids(self, test) -> List[str]:
        """Returns all prompt UIDs queue"""
        # TODO: Implement cache.
        return [item["prompt_id"] for item in self.query("queuing item", test=test)]

    def sut_response_prompt_uids_for_test(self, sut, test) -> List[str]:
        cached_responses = self.query("using cached sut response", sut=sut, test=test)
        fetched_responses = self.query("fetched sut response", sut=sut, test=test)
        all_prompts = [response["prompt_id"] for response in cached_responses + fetched_responses]
        return all_prompts


class JournalCheck(ABC):
    """All checks must inherit from this class."""

    @abstractmethod
    def check(self) -> bool:
        pass

    @abstractmethod
    def failure_message(self) -> str:
        """The message to display if the check fails."""
        pass


# TODO:
# class NumPromptsQueuedMatchesExpected(JournalCheck):
#     def __init__(self, search_engine: JournalSearch, sut, test):
#         # Load all data needed for the check.
#         self.num_test_prompts = search_engine.num_test_prompts(test)


class OneToOneCheck(JournalCheck):
    """Checks for a one-to-one mapping between two lists of prompt uids."""

    def __init__(self, expected_prompts: List[str], found_prompts: List[str]):
        found_counts = Counter(found_prompts)
        # TODO: Could probably make these 3 checks more efficient.
        self.duplicates = [uid for uid, count in found_counts.items() if count > 1]
        # Check for differences in the two sets.
        expected_prompts = set(expected_prompts)
        found_prompts = set(found_prompts)
        self.missing_prompts = list(expected_prompts - found_prompts)
        self.unknown_prompts = list(found_prompts - expected_prompts)

    def check(self) -> bool:
        return not any([len(self.duplicates), len(self.missing_prompts), len(self.unknown_prompts)])

    def failure_message(self) -> str:
        assert not self.check()
        messages = []
        if len(self.duplicates) > 0:
            messages.append(f"The following duplicate prompts were found: {self.duplicates}")
        if len(self.missing_prompts) > 0:
            messages.append(f"The prompts were expected but missing: {self.missing_prompts}")
        if len(self.unknown_prompts) > 0:
            messages.append(f"The following prompts were found but were not expected: {self.unknown_prompts}")
        return "\n\t".join(messages)


class EachPromptQueuedOnce(OneToOneCheck):
    def __init__(self, search_engine: JournalSearch, sut, test):
        queued_sut_entries = search_engine.query("queuing item", test=test, sut=sut)
        queued_sut_prompts = [entry["prompt_id"] for entry in queued_sut_entries]
        super().__init__(search_engine.test_prompt_uids(test), queued_sut_prompts)

    def failure_message(self) -> str:
        message = "Expected each prompt uid in the test to be queued exactly once.\n\t"
        # Call super() to get specific details about duplicates/missing/extra prompts.
        return message + super().failure_message()


class EachPromptRespondedToOnce(OneToOneCheck):
    def __init__(self, search_engine: JournalSearch, sut, test):
        super().__init__(
            search_engine.test_prompt_uids(test), search_engine.sut_response_prompt_uids_for_test(sut, test)
        )

    def failure_message(self) -> str:
        message = "Expected exactly 1 SUT response for each prompt in the test.\n\t"
        # Call super() to get specific details about duplicates/missing/extra prompts.
        return message + super().failure_message()


class EachResponseTranslatedOnce(OneToOneCheck):
    def __init__(self, search_engine: JournalSearch, sut, test):
        all_sut_response_prompts = search_engine.sut_response_prompt_uids_for_test(sut, test)
        translated_responses = search_engine.query("translated sut response", sut=sut, test=test)
        super().__init__(all_sut_response_prompts, [response["prompt_id"] for response in translated_responses])

    def failure_message(self) -> str:
        message = "Expected each SUT response to be translated exactly once.\n\t"
        # Call super() to get specific details about duplicates/missing/extra prompts.
        return message + super().failure_message()


class EachItemMeasuredOnce(OneToOneCheck):
    def __init__(self, search_engine: JournalSearch, sut, test):
        translated_responses = search_engine.query("translated sut response", sut=sut, test=test)
        measured_items = search_engine.query("measured item quality", sut=sut, test=test)
        super().__init__(
            [response["prompt_id"] for response in translated_responses],
            [response["prompt_id"] for response in measured_items],
        )

    def failure_message(self) -> str:
        message = "Expected every prompt-response to be measured exactly once.\n"
        # Call super() to get specific details about duplicates/missing/extra prompts.
        return message + super().failure_message()


class EachResponseAnnotatedOnce(OneToOneCheck):
    def __init__(self, search_engine: JournalSearch, sut, test, annotator):
        self.annotator = annotator
        self.sut = sut
        self.test = test
        translated_responses = search_engine.query("translated sut response", sut=sut, test=test)
        cached_annotations = search_engine.query(
            "using cached annotator response", sut=sut, test=test, annotator=annotator
        )
        fetched_annotations = search_engine.query("fetched annotator response", sut=sut, test=test, annotator=annotator)
        all_prompts_annotated = [response["prompt_id"] for response in cached_annotations + fetched_annotations]
        super().__init__([response["prompt_id"] for response in translated_responses], all_prompts_annotated)

    def failure_message(self) -> str:
        message = f"Expected exactly 1 {self.annotator} annotation for each response from SUT {self.sut} in test {self.test}\n\t"
        # Call super() to get specific details about duplicates/missing/extra prompts.
        return message + super().failure_message()


class EachAnnotationTranslatedOnce(OneToOneCheck):
    def __init__(self, search_engine: JournalSearch, sut, test, annotator):
        self.annotator = annotator
        self.sut = sut
        self.test = test
        cached_annotations = search_engine.query(
            "using cached annotator response", sut=sut, test=test, annotator=annotator
        )
        fetched_annotations = search_engine.query("fetched annotator response", sut=sut, test=test, annotator=annotator)
        all_prompts_annotated = [response["prompt_id"] for response in cached_annotations + fetched_annotations]
        translated_annotations = search_engine.query("translated annotation", sut=sut, test=test, annotator=annotator)

        super().__init__(all_prompts_annotated, [response["prompt_id"] for response in translated_annotations])

    def failure_message(self) -> str:
        message = f"Expected each {self.annotator} annotation for SUT {self.sut} in test {self.test} to be translated exactly once.\n\t"
        # Call super() to get specific details about duplicates/missing/extra prompts.
        return message + super().failure_message()


class JournalEntityLevelCheck:
    """A group of a checks that occupy the same conceptual entity-level in the journal.

    All checks in a group must accept the same entities in their init. params."""

    def __init__(self, name, check_classes, **entity_sets):
        """Each entity_set kwarg is a list of a type of entity."""
        self.name = name
        self.check_classes: List = check_classes
        # Outer-level dictionary keys are the entity tuples, inner dict. keys are the check names.
        # Values are boolean check results.
        self.results: Dict[str, Dict[str, bool | None]] | None = None
        self.row_names = None
        self.check_names = None
        self._init_results_table(**entity_sets)
        # List of warning messages for failed checks.
        self.warnings: List[str] = []

    def _init_results_table(self, **entity_sets):
        # Create an empty table where each row is an entity (or entity tuple) and each column is a check.
        self.results = defaultdict(dict)
        self.entity_names = sorted(list(entity_sets.keys()))
        self.row_names = []
        self.check_names = []

        for col in self.check_classes:
            self.check_names.append(self._col_name(col))
        for entity_tuple in product(*entity_sets.values()):
            entity_dict = dict(zip(entity_sets.keys(), entity_tuple))
            row_key = self._row_key(**entity_dict)
            self.row_names.append(row_key)
            # Each check is initialized to None to indicate it hasn't been run yet.
            self.results[row_key] = {col: None for col in self.check_names}

    @staticmethod
    def _row_key(**entities) -> str:
        """Return string key for a given set of entities."""
        sorted_keys = sorted(entities.keys())
        return ", ".join([entities[k] for k in sorted_keys])

    @staticmethod
    def _col_name(check_cls) -> str:
        return check_cls.__name__

    def check_is_complete(self) -> bool:
        """Make sure table is fully populated."""
        for row in self.row_names:
            for check in self.check_names:
                if self.results[row][check] is None:
                    return False
        return True

    def run_checks_for_row(self, search_engine, **entities):
        """Run all individual checks on a given entity tuple and store results and warnings."""
        for check_cls in self.check_classes:
            check = check_cls(search_engine, **entities)
            result = check.check()
            self.results[self._row_key(**entities)][self._col_name(check_cls)] = result
            if not result:
                # TODO: Add check name to warning message.
                self.warnings.append(f"{self._col_name(check_cls)}: {check.failure_message()}")


class ConsistencyChecker:

    def __init__(self, journal_path):
        self.journal_path = journal_path

        # Entities to run checks for.
        self.suts = None
        self.tests = None
        self.annotators = None
        self._collect_entities()

        # Checks to run at each level.
        self.test_sut_level_checker = JournalEntityLevelCheck(
            "Test x SUT level checks",
            [EachPromptQueuedOnce, EachPromptRespondedToOnce, EachResponseTranslatedOnce, EachItemMeasuredOnce],
            tests=self.tests,
            suts=self.suts,
        )
        self.test_sut_annotator_level_checker = JournalEntityLevelCheck(
            "Test x SUT x Annotator checks",
            [EachResponseAnnotatedOnce, EachAnnotationTranslatedOnce],
            tests=self.tests,
            suts=self.suts,
            annotators=self.annotators,
        )

    @property
    def _check_groups(self):
        """List of all sub-checkers."""
        return [self.test_sut_level_checker, self.test_sut_annotator_level_checker]

    def _collect_entities(self):
        # Get all SUTs and tests that were ran in the journal. We will run checks for each (SUT, test) pair.
        search_engine = JournalSearch(self.journal_path)
        starting_run_entry = search_engine.query("starting run")
        assert len(starting_run_entry) == 1

        self.suts = starting_run_entry[0]["suts"]
        self.tests = starting_run_entry[0]["tests"]
        # TODO: This assumes that all tests use the same annotators! Which is fine for now but may not hold-up later on.
        try:
            secrets = load_secrets_from_config()
            test_obj = TESTS.make_instance(self.tests[0], secrets=secrets)
            self.annotators = test_obj.get_annotators()
        except Exception as e:
            # Can't load test object, get annotators from journal instead.
            print("Failed to load test object. Collecting annotator UIDs to check from journal instead.")
            fetched_annotator_entries = search_engine.query("fetched annotator response", test=self.tests[0], sut=self.suts[0])
            cached_annotator_entries = search_engine.query("using cached annotator response", test=self.tests[0], sut=self.suts[0])
            self.annotators = list(set([entry["annotator"] for entry in fetched_annotator_entries + cached_annotator_entries]))

    def run(self, verbose=False):
        self._collect_results()
        self.display_results()
        if verbose:
            self.display_warnings()
        # TODO: Also run checks for the json record file.

    def _collect_results(self):
        """Populate the results/warning tables of each check level."""
        search_engine = JournalSearch(self.journal_path)
        for test in self.tests:
            for sut in self.suts:
                self.test_sut_level_checker.run_checks_for_row(search_engine, sut=sut, test=test)
                for annotator in self.annotators:
                    self.test_sut_annotator_level_checker.run_checks_for_row(
                        search_engine, sut=sut, test=test, annotator=annotator
                    )

    @staticmethod
    def format_result(result: bool):
        return "✅" if result else "❌"

    def checks_are_complete(self) -> bool:
        for checker in self._check_groups:
            if not checker.check_is_complete():
                return False
        return True

    def checks_all_passed(self) -> bool:
        assert self.checks_are_complete(), "Cannot determine pass/fail for this journal until all checks have been run."
        for checker in self._check_groups:
            if any(not result for results in checker.results.values() for result in results.values()):
                return False
        return True

    def display_results(self):
        """Print simple table where each row is a single entity (or entity tuple e.g. test x SUT) and each column is a check."""
        assert self.checks_are_complete(), "Cannot display results until all checks have been run."
        for checker in self._check_groups:
            print("Results for", checker.name)
            results_table = []
            for entity, checks in checker.results.items():
                results_table.append([entity] + [self.format_result(checks[c]) for c in checker.check_names])
            print(tabulate(results_table, headers=[", ".join(checker.entity_names)] + list(checker.check_names)))
            print()

    def display_warnings(self):
        """Print details about the failed checks."""
        assert self.checks_are_complete(), "Cannot display results until all checks have been run."
        for checker in self._check_groups:
            print("-" * LINE_WIDTH)
            if len(checker.warnings) == 0:
                print(f"All {checker.name} checks passed!")
                return
            print(f"Failed checks for {checker.name}:")
            for warning in checker.warnings:
                print(warning)  # or something


def summarize_consistency_check_results(checkers: List[ConsistencyChecker]):
    """Print a table summarizing the overall pass/fail results for multiple consistency checks."""
    summary_table = []
    for checker in checkers:
        if checker.checks_are_complete():
            result = ConsistencyChecker.format_result(checker.checks_all_passed())
        else:
            result = "INCOMPLETE"
        summary_table.append([checker.journal_path, result])

    print(tabulate(summary_table, headers=["Journal", "All checks passed"]))
