from abc import ABC, abstractmethod
from typing import List, Union

from newhelm.placeholders import Measurement, Prompt, Result
from newhelm.single_turn_prompt_response import (
    AnnotatedTestItem,
    MeasuredTestItem,
    TestItem,
)


class BaseTest(ABC):
    """This is the placeholder base class for all tests."""

    pass


class BasePromptResponseTest(BaseTest, ABC):
    """This is the base class for all tests that are single turn."""

    @abstractmethod
    def make_test_items(self) -> List[TestItem]:
        """Generate all data that will eventually go to the SUT."""
        pass

    # TODO Insert a method here for how the test can specify what annotators to run.

    @abstractmethod
    def measure_quality(self, item: AnnotatedTestItem) -> List[Measurement]:
        """Use the SUT responses with annotations to determine how well the SUT did on this TestItem."""
        pass

    @abstractmethod
    def aggregate_measurements(self, items: List[MeasuredTestItem]) -> List[Result]:
        """Combine the measurements for each TestItem into a list of Results."""
        pass
