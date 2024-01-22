from dataclasses import dataclass
from typing import Dict, List, Mapping
from newhelm.annotation import Annotation
from typing import Dict, List, Mapping
from newhelm.annotation import Annotation

from newhelm.benchmark import Score
from newhelm.placeholders import Result
from newhelm.single_turn_prompt_response import (
    PromptInteraction,
    TestItem,
)


@dataclass(frozen=True)
class TestItemRecord:
    # TODO: This duplicates the list of prompts across test_item and interactions.
    # Maybe just copy the TestItem context.
    test_item: TestItem
    interactions: List[PromptInteraction]
    annotations: Dict[str, Annotation]
    measurements: Dict[str, float]


@dataclass(frozen=True)
class TestRecord:
    """This is a rough sketch of the kind of data we'd want every Test to record."""

    test_name: str
    dependency_versions: Mapping[str, str]
    sut_name: str
    # TODO We should either reintroduce "Turns" here, or expect
    # there to b different schemas for different TestImplementationClasses.
    test_item_records: List[TestItemRecord]
    results: List[Result]


@dataclass(frozen=True)
class BenchmarkRecord:
    """This is a rough sketch of the kind of data we'd want every Benchmark to record."""

    benchmark_name: str
    sut_name: str
    test_records: List[TestRecord]
    score: Score
