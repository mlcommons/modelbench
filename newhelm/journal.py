from dataclasses import dataclass
from typing import List, Mapping

from newhelm.benchmark import Score
from newhelm.placeholders import Result
from newhelm.single_turn_prompt_response import AnnotatedTestItem


@dataclass(frozen=True)
class TestJournal:
    """This is a rough sketch of the kind of data we'd want every Test to record."""

    test_name: str
    dependency_versions: Mapping[str, str]
    sut_name: str
    # TODO We should either reintroduce "Turns" here, or expect
    # there to b different schemas for different TestImplementationClasses.
    annotated_interactions: List[AnnotatedTestItem]
    results: List[Result]


@dataclass(frozen=True)
class BenchmarkJournal:
    """This is a rough sketch of the kind of data we'd want every Benchmark to record."""

    benchmark_name: str
    sut_name: str
    test_journals: List[TestJournal]
    score: Score
