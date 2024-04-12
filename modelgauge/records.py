from modelgauge.base_test import TestResult
from modelgauge.general import current_local_datetime
from modelgauge.record_init import InitializationRecord
from modelgauge.single_turn_prompt_response import (
    PromptInteractionAnnotations,
    TestItem,
)
from pydantic import AwareDatetime, BaseModel, Field
from typing import Dict, List, Mapping


class TestItemRecord(BaseModel):
    """Record of all data relevant to a single TestItem."""

    # TODO: This duplicates the list of prompts across test_item and interactions.
    # Maybe just copy the TestItem context.
    test_item: TestItem
    interactions: List[PromptInteractionAnnotations]
    measurements: Dict[str, float]

    __test__ = False


class TestRecord(BaseModel):
    """Record of all data relevant to a single run of a Test."""

    run_timestamp: AwareDatetime = Field(default_factory=current_local_datetime)
    test_uid: str
    test_initialization: InitializationRecord
    dependency_versions: Mapping[str, str]
    sut_uid: str
    sut_initialization: InitializationRecord
    # TODO We should either reintroduce "Turns" here, or expect
    # there to b different schemas for different TestImplementationClasses.
    test_item_records: List[TestItemRecord]
    result: TestResult

    __test__ = False
