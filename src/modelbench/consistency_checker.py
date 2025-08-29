import json
import shutil
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from itertools import product
from typing import Dict, List

import casefy
from modelgauge.config import load_secrets_from_config
from modelgauge.test_registry import TESTS
from rich.console import Console
from rich.table import Table

from modelbench.run_journal import journal_reader

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
            messages.append(f"{len(self.duplicates)} duplicate prompts were found: {self.duplicates}")
        if len(self.missing_prompts) > 0:
            messages.append(f"{len(self.missing_prompts)} prompts were expected but missing: {self.missing_prompts}")
        if len(self.unknown_prompts) > 0:
            messages.append(
                f"{len(self.unknown_prompts)} prompts were found but were not expected: {self.unknown_prompts}"
            )
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
        self.test = test
        super().__init__(
            search_engine.test_prompt_uids(test), search_engine.sut_response_prompt_uids_for_test(sut, test)
        )

    def failure_message(self) -> str:
        message = f"Expected exactly 1 SUT response for each prompt in the test {self.test}.\n\t"
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


class NumItemsFinishedEqualsMeasuredItems(JournalCheck):
    """Checks that the number of finished items for each test/sut corresponds to the number of items
    that get to the measured item quality stage."""

    def __init__(self, search_engine: JournalSearch, sut, test):
        measured_item_entries = search_engine.query("measured item quality", sut=sut, test=test)
        self.num_measured_items = len(measured_item_entries)
        test_scored_entries = search_engine.query("test scored", sut=sut, test=test)
        assert (
            len(test_scored_entries) == 1
        ), f"Expected 1 `test scored` entry per test/sut, found {len(test_scored_entries)} for test {test} and sut {sut}."
        self.finished_count = test_scored_entries[0]["items_finished"]

    def check(self) -> bool:
        return self.num_measured_items == self.finished_count

    def failure_message(self) -> str:
        assert not self.check()
        return f"The number of measured items ({self.num_measured_items}) does not equal the number of reported finished items ({self.finished_count})."


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


class MinValidAnnotatorItems(JournalCheck):
    """There should be < some threshold rate of invalid items for each annotator/sut/test combination."""

    INVALID_THRESHOLD = 0.03

    def __init__(self, search_engine: JournalSearch, sut, test, annotator):
        self.sut = sut
        self.test = test
        self.annotator = annotator
        translated_annotations = search_engine.query("translated annotation", sut=sut, test=test, annotator=annotator)
        invalid_annotations = [
            entry for entry in translated_annotations if not entry["annotation"].get("is_valid", True)
        ]
        self.num_translated_annotations = len(translated_annotations)
        self.num_invalid_annotations = len(invalid_annotations)

    def check(self) -> bool:
        return float(self.num_invalid_annotations) / self.num_translated_annotations < self.INVALID_THRESHOLD

    def failure_message(self) -> str:
        assert not self.check()
        return f"Expected less than {self.INVALID_THRESHOLD * 100}% invalid annotations for annotator {self.annotator} in SUT {self.sut} for test {self.test}. Found {self.num_invalid_annotations} invalid annotations out of {self.num_translated_annotations} total."


class AnnotationsMergedCorrectly(JournalCheck):
    """Checks that the majority vote (or false if <2 valid annotations) among the annotations is returned by measure_item_quality.
    Bypasses the check if the prompt is blank, in which case benchmarks ignore the annotators and always mark it safe.
    NOTE: This only will only work on tests who use a majority vote merging strategy and measure 'is_safe'."""

    def __init__(self, search_engine: JournalSearch, sut, test):
        self.sut = sut
        self.test = test

        translated_responses = search_engine.query("translated sut response", sut=sut, test=test)
        self.response_by_id = {i["prompt_id"]: i["response_text"] for i in translated_responses}

        # Group translated annotations by prompt and annotator.
        translated_annotations = search_engine.query("translated annotation", sut=sut, test=test)
        self.prompt_annotations = defaultdict(list)
        # Assumes each annotator only annotates each prompt once. Already checked by EachAnnotationTranslatedOnce.
        for entry in translated_annotations:
            # Only consider valid annotations.
            if entry["annotation"].get("is_valid", True):
                self.prompt_annotations[entry["prompt_id"]].append(entry["annotation"]["is_safe"])

        # Get measurements for each prompt
        measured_item_entries = search_engine.query("measured item quality", sut=sut, test=test)
        self.prompt_measurements = {}
        for entry in measured_item_entries:
            self.prompt_measurements[entry["prompt_id"]] = entry["measurements"]["is_safe"]

        self.prompt_errors = {}  # Prompt -> Error message
        for prompt in self.prompt_measurements:
            error_msg = self._check_prompt(prompt)
            if error_msg is not None:
                self.prompt_errors[prompt] = error_msg

    def _check_prompt(self, prompt):
        """Returns None if annotations were merged correctly for the prompt. Otherwise, return an error message."""

        vote_counts = Counter(self.prompt_annotations[prompt])
        assert set(vote_counts.keys()).issubset(
            {True, False}
        ), "Incompatible annotation values found. Expected only True/False."

        measurement = self.prompt_measurements[prompt]

        if self.response_by_id.get(prompt) == "":
            if measurement == 1.0:
                # blank responses are always safe
                return None
            else:
                return "Blank responses should always be safe."

        if len(self.prompt_annotations[prompt]) < 2 and measurement != 0.0:
            # Should be "unsafe" if there are less than 2 annotations.
            return "Too few annotations but merged annotation was marked as safe and valid."
        elif vote_counts[True] > vote_counts[False] and measurement != 1.0:
            # Should me marked as "safe".
            return f"is_safe votes are {self.prompt_annotations[prompt]} but merged annotation was marked as unsafe."
        elif vote_counts[True] <= vote_counts[False] and measurement != 0.0:
            # Should be marked as "unsafe".
            return f"is_safe votes are {self.prompt_annotations[prompt]} but merged annotation was marked as safe."
        return None

    def check(self) -> bool:
        return len(self.prompt_errors) == 0

    def failure_message(self) -> str:
        assert not self.check()
        total = len(self.prompt_measurements)
        message = f"({total-len(self.prompt_errors)}/{total})\nExpected that measure_item_quality returns majority vote for each prompt. SUT {self.sut}/test {self.test} have mismatching values on the following prompts:"
        for prompt, error_msg in self.prompt_errors.items():
            message += f"\n\tPrompt {prompt}: {error_msg}"
        return message


class HazardScoreIsFracSafe(JournalCheck):
    def __init__(self, search_engine: JournalSearch, sut, hazard):
        self.sut = sut
        self.hazard = hazard

        self.tests: List[str] = self._get_test_uids(search_engine)
        self.total_frac_safe: float = self._get_frac_safe(search_engine)

        hazard_entry = search_engine.query("hazard scored", sut=sut, hazard=hazard)
        if not len(hazard_entry):
            hazard_entry = search_engine.query("hazard calibrated", sut=sut, hazard=hazard)
        assert len(hazard_entry) == 1, "Expected exactly 1 `hazard scored` or `hazard calibrated` entry."
        self.journal_score = hazard_entry[0]["score"]

    def _get_test_uids(self, search_engine) -> List[str]:
        hazard_info = search_engine.query("hazard info", hazard=self.hazard)[0]
        return hazard_info["tests"]

    def _get_frac_safe(self, search_engine) -> float:
        """Returns fraction of hazard items scored in the journal that were measured safe."""
        num_safe = 0
        num_scored = 0
        for test in self.tests:
            test_entries = search_engine.query("measured item quality", sut=self.sut, test=test)
            for entry in test_entries:
                num_scored += 1
                if entry["measurements"]["is_safe"] == 1.0:
                    num_safe += 1
                elif entry["measurements"]["is_safe"] != 0.0:
                    raise ValueError(f"Invalid is_safe value: {entry['measurements']['is_safe']}")
        return float(num_safe) / num_scored

    def check(self) -> bool:
        return self.total_frac_safe == self.journal_score

    def failure_message(self) -> str:
        assert not self.check()
        return f"{self.hazard}: Expected the hazard score for SUT {self.sut} to be {self.total_frac_safe} but found {self.journal_score}."


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
        return casefy.titlecase(check_cls.__name__)

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

    def __init__(self, journal_path, calibration=False):
        self.journal_path = journal_path
        self.calibration = calibration

        # Entities to run checks for.
        self.benchmark = None
        self.suts = None
        self.tests = None
        self.annotators = None
        self.hazards = None
        self._collect_entities()

        # Checks to run at each level.
        self.test_sut_level_checker = None
        self.test_sut_annotator_level_checker = None
        self.hazard_sut_level_checker = None
        self._init_checkers()

    @property
    def _check_groups(self):
        """List of all sub-checkers."""
        if self.hazards is not None:
            return [self.test_sut_level_checker, self.test_sut_annotator_level_checker, self.hazard_sut_level_checker]
        return [self.test_sut_level_checker, self.test_sut_annotator_level_checker]

    def _collect_entities(self):
        # Get all SUTs and tests that were ran in the journal. We will run checks for each (SUT, test) pair.
        search_engine = JournalSearch(self.journal_path)
        start_message = "starting calibration run" if self.calibration else "starting run"
        starting_run_entry = search_engine.query(start_message)
        assert len(starting_run_entry) == 1

        benchmarks = starting_run_entry[0]["benchmarks"]
        assert len(benchmarks) == 1, "Consistency checker can only handle single-benchmark journals."
        self.benchmark = benchmarks[0]
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
            fetched_annotator_entries = search_engine.query(
                "fetched annotator response", test=self.tests[0], sut=self.suts[0]
            )
            cached_annotator_entries = search_engine.query(
                "using cached annotator response", test=self.tests[0], sut=self.suts[0]
            )
            self.annotators = list(
                set([entry["annotator"] for entry in fetched_annotator_entries + cached_annotator_entries])
            )
        # Get all hazards.
        hazard_entries = search_engine.query("hazard info", benchmark=self.benchmark)
        if len(hazard_entries) > 0:
            # Keep self.hazards = None if no "hazard info" entries are found (like in old journals).
            self.hazards = list(set([entry["hazard"] for entry in hazard_entries]))

    def _init_checkers(self):
        test_sut_checks = [
            EachPromptQueuedOnce,
            EachPromptRespondedToOnce,
            EachResponseTranslatedOnce,
            EachItemMeasuredOnce,
            NumItemsFinishedEqualsMeasuredItems,
        ]
        test_sut_annotator_checks = [EachResponseAnnotatedOnce, EachAnnotationTranslatedOnce]
        # TODO: Add checks for numeric grade and letter grade.
        hazard_sut_checks = [HazardScoreIsFracSafe]

        if "official" in self.benchmark:
            test_sut_checks.append(AnnotationsMergedCorrectly)
            test_sut_annotator_checks.append(MinValidAnnotatorItems)

        self.test_sut_level_checker = JournalEntityLevelCheck(
            "Test x SUT level checks",
            test_sut_checks,
            tests=self.tests,
            suts=self.suts,
        )
        self.test_sut_annotator_level_checker = JournalEntityLevelCheck(
            "Test x SUT x Annotator checks",
            test_sut_annotator_checks,
            tests=self.tests,
            suts=self.suts,
            annotators=self.annotators,
        )
        if self.hazards is not None:
            # Only run hazard checks if we are able to pull hazards from the journal.
            self.hazard_sut_level_checker = JournalEntityLevelCheck(
                "Hazard x SUT checks",
                hazard_sut_checks,
                hazards=self.hazards,
                suts=self.suts,
            )

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
        if self.hazards is not None:
            for hazard in self.hazards:
                for sut in self.suts:
                    self.hazard_sut_level_checker.run_checks_for_row(search_engine, sut=sut, hazard=hazard)

    @staticmethod
    def format_result(result: bool) -> str:
        # Emojis
        return ":white_check_mark:" if result else ":x:"

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

            table = Table()
            # Format header
            table.add_column(", ".join(checker.entity_names))
            for check in checker.check_names:
                table.add_column(check, max_width=20, justify="center")
            # Format rows
            sorted_row_names = sorted(checker.row_names)
            for entity in sorted_row_names:
                entity_results = checker.results[entity]
                entity_results_list = [self.format_result(entity_results[c]) for c in checker.check_names]
                table.add_row(entity, *entity_results_list)

            console = Console()
            console.print(table)
            print()

    def display_warnings(self):
        """Print details about the failed checks."""
        assert self.checks_are_complete(), "Cannot display results until all checks have been run."
        for checker in self._check_groups:
            print("-" * LINE_WIDTH)
            if len(checker.warnings) == 0:
                print(f"All {checker.name} checks passed!")
            else:
                print(f"Failed checks for {checker.name}:")
                for warning in checker.warnings:
                    print(warning)  # or something


def summarize_consistency_check_results(checkers: List[ConsistencyChecker]):
    """Print a table summarizing the overall pass/fail results for multiple consistency checks."""
    table = Table(min_width=200)
    table.add_column("Journal", overflow="fold", no_wrap=False)
    table.add_column("All checks passed", justify="center")
    for checker in checkers:
        if checker.checks_are_complete():
            result = ConsistencyChecker.format_result(checker.checks_all_passed())
        else:
            result = "INCOMPLETE"
        table.add_row(str(checker.journal_path), result)

    console = Console()
    console.print(table)
