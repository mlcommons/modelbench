import json
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import product
from tabulate import tabulate
from typing import Dict, List

from modelbench.run_journal import journal_reader


class JournalSearch:
    def __init__(self, journal_path, record_path):
        self.journal_path = journal_path
        self.record_path = record_path
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


class JournalCheck(ABC):
    """All checks must inherit from this class + their respective entity level class."""

    # @abstractmethod
    # def required_messages(self) -> List[str]:
    #     """The set of messages that this check relies on.
    #     Might be used to filter out irrelevant journal entries. not sure if that would actually help."""
    #     pass

    @abstractmethod
    def check(self) -> bool:
        pass

    @abstractmethod
    def failure_message(self) -> str:
        """The message to display if the check fails."""
        pass


class EachPromptHasOneResponse(JournalCheck):
    def __init__(self, search_engine: JournalSearch, sut, test):
        # Load all data needed for the check.
        self.num_cached_responses = len(search_engine.query("using cached sut response", sut=sut, test=test))
        self.num_fetched_responses = len(search_engine.query("fetched sut response", sut=sut, test=test))
        # Get num. test prompts
        test_entry = search_engine.query("using test items", test=test)
        assert len(test_entry) == 1
        self.num_test_prompts = test_entry[0]["using"]

    def check(self) -> bool:
        return self.num_cached_responses + self.num_fetched_responses == self.num_test_prompts

    def failure_message(self) -> str:
        """The message to display if the check fails."""
        assert not self.check()
        return f"The total number of SUT responses (cached=({self.num_cached_responses} + fetched = {self.num_fetched_responses}) does not correspond to the total num prompts in test ({self.num_test_prompts})"


class EachResponseTranslatedOnce(JournalCheck):
    def __init__(self, search_engine: JournalSearch, sut, test):
        # Load all data needed for the check.
        self.num_translated_responses = len(search_engine.query("translated sut response", sut=sut, test=test))
        self.num_test_prompts = search_engine.query("using test items", test=test)[0]["using"]

    def check(self) -> bool:
        return self.num_translated_responses == self.num_test_prompts

    def failure_message(self) -> str:
        """The message to display if the check fails."""
        assert not self.check()
        return f"The number of SUT response translated ({self.num_translated_responses}) does not equal the total num prompts in test ({self.num_test_prompts})"


class JournalEntityLevelCheck:
    """A group of a checks that occupy the same conceptual entity-level in the journal.

    All checks in a group must accept the same entities in their init. params."""

    def __init__(self, check_classes, **entity_sets):
        """Each entity_set kwarg is a list of a type of entity."""
        self.check_classes: List = check_classes
        # Outer-level dictionary keys are the entity tuples, inner dict. keys are the check names.
        # Values are boolean check results.
        self.results: Dict[str, Dict[str, bool | None]] | None = None
        self.row_names = None
        self.check_names = None
        self._init_results_table(**entity_sets)
        # List of warning messages for failed checks.
        self.warnings: List[str] = []

    def _init_results_table(self, **entity_sets) -> Dict[str, Dict[str, bool | None]]:
        # Create an empty table where each row is an entity (or entity tuple) and each column is a check.
        self.results = defaultdict(dict)
        self.row_names = []
        self.check_names = []

        for col in self.check_classes:
            self.check_names.append(self._col_name(col))
        for entity_tuple in product(*entity_sets.values()):
            row_key = self._row_key(entity_tuple)
            self.row_names.append(row_key)
            # Each check is initialized to None to indicate it hasn't been run yet.
            self.results[row_key] = {col: None for col in self.check_names}

    @staticmethod
    def _row_key(*entities) -> str:
        """Return string key for a given set of entities."""
        return ", ".join(sorted(*entities))

    @staticmethod
    def _col_name(check_cls) -> str:
        return check_cls.__name__

    def check_is_complete(self):
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
            self.results[self._row_key(entities.values())][self._col_name(check_cls)] = result
            if not result:
                self.warnings.append(check.failure_message())


class ConsistencyChecker:

    def __init__(self, journal_path, record_path):
        # Object holding journal entries
        self.search_engine = JournalSearch(journal_path, record_path)

        # Entities to run checks for.
        self.suts = None
        self.tests = None
        self.annotators = None
        self._collect_entities()

        # Checks to run at each level.
        self.test_sut_level_checker = JournalEntityLevelCheck(
            [EachPromptHasOneResponse, EachResponseTranslatedOnce], tests=self.tests, suts=self.suts
        )
        self.test_sut_annotator_level_checker = JournalEntityLevelCheck([])  # TODO

    def _collect_entities(self):
        # Get all SUTs and tests that were ran in the journal. We will run checks for each (SUT, test) pair.
        starting_run_entry = self.search_engine.query("starting run")
        assert len(starting_run_entry) == 1

        self.suts = starting_run_entry[0]["suts"]
        self.tests = starting_run_entry[0]["tests"]
        # TODO: Get annotators
        self.annotators = []

    def run(self, verbose=False):
        self.collect_results()
        self.display_results()
        if verbose:
            self.display_warnings()
        # TODO: Also run checks for the json record file.

    def collect_results(self):
        """Populate the results/warning tables of each check level."""
        for test in self.tests:
            for sut in self.suts:
                self.test_sut_level_checker.run_checks_for_row(self.search_engine, sut=sut, test=test)
                for annotator in self.annotators:
                    self.test_sut_annotator_level_checker.run_checks_for_row(
                        self.search_engine, sut=sut, test=test, annotator=annotator
                    )

    def display_results(self):
        """Print simple table where each row is a single entity (or entity tuple e.g. test x SUT) and each column is a check."""
        check_groups = [self.test_sut_level_checker, self.test_sut_annotator_level_checker]
        for checker in check_groups:
            print("Results for", checker.__class__.__name__)
            assert checker.check_is_complete()
            results_table = []
            for entity, checks in checker.results.items():
                results_table.append([entity] + [checks[c] for c in checker.check_names])
            print(tabulate(results_table, headers=["Entity"] + list(checker.check_names)))
            print()

    def display_warnings(self):
        """Print details about the failed checks."""
        check_groups = [self.test_sut_level_checker, self.test_sut_annotator_level_checker]
        for checker in check_groups:
            assert checker.check_is_complete()
            if len(checker.warnings) == 0:
                print(f"All {checker.__class__.__name__} checks passed!")
                return
            print(f"Failed checks for {checker.__class__.__name__}:")
            for warning in checker.warnings:
                print(warning)  # or something
