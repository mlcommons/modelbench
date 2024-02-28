import datetime
from typing import Dict, List, Mapping

from pydantic import AwareDatetime, BaseModel, Field
from newhelm.annotation import Annotation
from typing import Dict, List, Mapping
from newhelm.annotation import Annotation
from newhelm.base_test import Result
from newhelm.general import current_local_datetime
from newhelm.record_init import InitializationRecord
from newhelm.single_turn_prompt_response import (
    PromptInteraction,
    TestItem,
)


class TestItemRecord(BaseModel):
    # TODO: This duplicates the list of prompts across test_item and interactions.
    # Maybe just copy the TestItem context.
    test_item: TestItem
    interactions: List[PromptInteraction]
    annotations: Dict[str, Annotation]
    measurements: Dict[str, float]

    __test__ = False


class TestRecord(BaseModel):
    """This is a rough sketch of the kind of data we'd want every Test to record."""

    run_timestamp: AwareDatetime = Field(default_factory=current_local_datetime)
    test_name: str
    test_initialization: InitializationRecord
    dependency_versions: Mapping[str, str]
    sut_name: str
    sut_initialization: InitializationRecord
    # TODO We should either reintroduce "Turns" here, or expect
    # there to b different schemas for different TestImplementationClasses.
    test_item_records: List[TestItemRecord]
    results: List[Result]

    __test__ = False
