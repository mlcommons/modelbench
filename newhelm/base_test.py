from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Mapping, Union
from newhelm.dependency_helper import DependencyHelper
from newhelm.external_data import ExternalData

from newhelm.placeholders import Result
from newhelm.single_turn_prompt_response import (
    TestItemAnnotations,
    MeasuredTestItem,
    TestItem,
)


@dataclass(frozen=True)
class TestMetadata:
    """Structured information about the Test which we can use in the UI to list available tests
    as well as for filtering to specific kinds of tests."""

    name: str
    description: str

    # Convince pytest to ignore this class.
    __test__ = False


class BaseTest(ABC):
    """This is the placeholder base class for all tests."""

    @classmethod
    @abstractmethod
    def get_metadata(cls) -> TestMetadata:
        """Return a description of the test."""
        pass


class BasePromptResponseTest(BaseTest, ABC):
    """This is the base class for all tests that are single turn."""

    @classmethod
    @abstractmethod
    def get_dependencies(cls) -> Mapping[str, ExternalData]:
        """Return a mapping of external dependency name to how it can be found downloaded."""
        pass

    @abstractmethod
    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        """Generate all data that will eventually go to the SUT."""
        pass

    # TODO Insert a method here for how the test can specify what annotators to run.

    @abstractmethod
    def measure_quality(self, item: TestItemAnnotations) -> Dict[str, float]:
        """Use the SUT responses with annotations to determine how well the SUT did on this TestItem."""
        pass

    @abstractmethod
    def aggregate_measurements(self, items: List[MeasuredTestItem]) -> List[Result]:
        """Combine the measurements for each TestItem into a list of Results."""
        pass
