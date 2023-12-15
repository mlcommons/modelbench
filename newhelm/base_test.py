from abc import ABC, abstractmethod
from typing import List
from newhelm.annotation import AnnotatedInteraction

from newhelm.placeholders import PromptTemplate, Result


class BaseTest(ABC):
    """This is the placeholder base class for all tests."""

    pass


class BasePromptResponseTest(BaseTest, ABC):
    """This is the base class for all tests that are single turn."""

    @abstractmethod
    def make_prompt_templates(self) -> List[PromptTemplate]:
        """Generate all data that will eventually go to the SUT."""
        pass

    # TODO Insert a method here for how the test can specify what annotators to run.

    @abstractmethod
    def calculate_results(
        self, interactions: List[AnnotatedInteraction]
    ) -> List[Result]:
        """Use the SUT responses with annotations to produce a list of Results."""
        pass
