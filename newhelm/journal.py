from dataclasses import dataclass
from typing import List

from newhelm.annotation import AnnotatedInteraction
from newhelm.benchmark import Score
from newhelm.placeholders import Result


@dataclass(frozen=True)
class TestJournal:
    """This is a rough sketch of the kind of data we'd want every Test to record."""

    test_name: str
    sut_name: str
    annotated_interactions: List[AnnotatedInteraction]
    results: List[Result]


@dataclass(frozen=True)
class BenchmarkJournal:
    """This is a rough sketch of the kind of data we'd want every Benchmark to record."""

    benchmark_name: str
    sut_name: str
    test_journals: List[TestJournal]
    score: Score
