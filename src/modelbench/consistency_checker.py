import json
from abc import ABC, abstractmethod
from collections import defaultdict
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

    @abstractmethod
    def required_messages(self) -> List[str]:
        """The set of messages that this check relies on.
        Might be used to filter out irrelevant journal entries. not sure if that would actually help."""
        pass

    @abstractmethod
    def check(self) -> bool:
        pass

    @abstractmethod
    def failure_message(self) -> str:
        """The message to display if the check fails."""
        pass


class EachPromptHasOneResponse(JournalCheck):
    def __init__(self, sut, test, search_engine: JournalSearch):
        # Load all data needed for the check.
        self.num_cached_responses = len(search_engine.query("using cached sut response", sut=sut, test=test))
        self.num_fetched_responses = len(search_engine.query("fetched sut response", sut=sut, test=test))
        # Get num. test prompts
        test_entry = search_engine.query("using test items", test=test)
        assert len(test_entry) == 1
        self.num_test_prompts = test_entry[0]["using"]

    def required_messages(self) -> List[str]:
        return ["using cached sut response", "fetched sut response", "using test items"]

    def check(self) -> bool:
        return self.num_cached_responses + self.num_fetched_responses == self.num_test_prompts

    def failure_message(self) -> str:
        """The message to display if the check fails."""
        assert not self.check()
        return f"The total number of SUT responses (cached=({self.num_cached_responses} + fetched = {self.num_fetched_responses}) does not correspond to the total num prompts in test ({self.num_test_prompts})"


class EachResponseTranslatedOnce(JournalCheck):
    def __init__(self, sut, test, search_engine: JournalSearch):
        # Load all data needed for the check.
        self.num_translated_responses = len(search_engine.query("translated sut response", sut=sut, test=test))
        self.num_test_prompts = search_engine.query("using test items", test=test)[0]["using"]

    def required_messages(self) -> List[str]:
        return ["translated sut response", "using test items"]

    def check(self) -> bool:
        return self.num_translated_responses == self.num_test_prompts

    def failure_message(self) -> str:
        """The message to display if the check fails."""
        assert not self.check()
        return f"The number of SUT response translated ({self.num_translated_responses}) does not equal the total num prompts in test ({self.num_test_prompts})"


class JournalEntityLevelCheck(ABC):
    """Concrete subclasses run all checks that occupy the same conceptual entity-level in the journal"""

    def __init__(self):
        """All subclasses must call super().__init__()."""
        self.results = defaultdict(dict)
        self.warnings = []

    @abstractmethod
    def collect_entities(self, search_engine: JournalSearch):
        """Collect all entities relevant to this level of checking from the journal.
        These entities will be used to run checks.
        """
        pass

    @abstractmethod
    def run_checks(self, search_engine: JournalSearch):
        """Store results in self.results and self.warnings."""
        pass

    def display_all_results(self):
        """Print simple table where each row is a single entity (or entity tuple e.g. test x SUT) and each column is a check."""
        # TODO
        # Not sure if this should be implemented here, by the concrete subclass, or in ConsistencyChecker...
        pass

    def display_warnings(self):
        """Print details about the failed checks."""
        for warning in self.warnings:
            print(warning)  # or something


class SUTxTestLevelChecker(JournalEntityLevelCheck):
    """Runs all checks relevant to a (SUT, test) tuple."""

    def __init__(self):
        super().__init__()
        self.suts = []
        self.tests = []
        # All checks must accept `sut` and `test` init. params.
        self.check_classes = [EachPromptHasOneResponse, EachResponseTranslatedOnce]

    def collect_entities(self, search_engine: JournalSearch):
        # Get all SUTs and tests that were ran in the journal. We will run checks for each (SUT, test) pair.
        starting_run_entry = search_engine.query("starting run")
        assert len(starting_run_entry) == 1
        self.suts = starting_run_entry[0]["suts"]
        self.tests = starting_run_entry[0]["tests"]

    def run_checks(self, search_engine: JournalSearch):
        for sut in self.suts:
            for test in self.tests:
                things = f"{sut}-{test}"
                for cls in self.check_classes:
                    check = cls(sut, test, search_engine)
                    self.results[things][cls] = check.check()  # True or False
                    if not self.results[things][cls]:
                        self.warnings.append(check.failure_message())


class ConsistencyChecker:

    def __init__(self, journal_path, record_path):
        # Get all subclasses of JournalEntityLevelCheck
        self.check_levels = [SUTxTestLevelChecker()]

        # Object holding journal entries
        self.search_engine = JournalSearch(journal_path, record_path)

    def run_checks(self, verbose=False):
        for level in self.check_levels:
            level.run_checks(self.search_engine)
            # TODO: Maybe move the display-formatting logic into this class.
            level.display_all_results()
            if verbose:
                level.display_warnings()
        # TODO: Also run checks for the json record file.
