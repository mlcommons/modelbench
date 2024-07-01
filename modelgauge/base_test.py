from abc import ABC, abstractmethod
from modelgauge.annotator import Annotator
from modelgauge.dependency_helper import DependencyHelper
from modelgauge.external_data import ExternalData
from modelgauge.record_init import InitializationRecord
from modelgauge.single_turn_prompt_response import (
    MeasuredTestItem,
    TestItem,
    TestItemAnnotations,
)
from modelgauge.sut_capabilities import SUTCapability
from modelgauge.tracked_object import TrackedObject
from modelgauge.typed_data import Typeable, TypedData
from typing import Dict, List, Mapping, Sequence, Type


class BaseTest(TrackedObject):
    """This is the placeholder base class for all tests.

    Test classes should be decorated with `@modelgauge_test`, which sets the
    class attribute `requires_sut_capabilities` as well as `initialization_record` of test instances.

    Attributes:
        requires_sut_capabilities: List of capabilities a SUT must report in order to run this test.
            Test classes must specify their requirements in the `@modelgauge_test` decorator args.
        uid (str): Unique identifier for a test instance.
        initialization_record: Initialization data that can be used to reconstruct a test instance.
    """

    # Set automatically by @modelgauge_test()
    requires_sut_capabilities: Sequence[Type[SUTCapability]]

    def __init__(self, uid: str):
        super().__init__(uid)
        # The initialization record is set automatically by @modelgauge_test()
        self.initialization_record: InitializationRecord


class PromptResponseTest(BaseTest, ABC):
    """Interface for all tests that are single turn.

    Concrete subclasses must implement every method in the interface.
    See `BaseTest` for more information regarding test implementation."""

    @abstractmethod
    def get_dependencies(self) -> Mapping[str, ExternalData]:
        """Return a mapping of external dependency name to how it can be found downloaded."""
        pass

    @abstractmethod
    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        """Generate all data that will eventually go to the SUT."""
        pass

    @abstractmethod
    def get_annotators(self) -> Mapping[str, Annotator]:
        """Return a mapping of annotators this Test wants to run.

        Mapping can be empty. Key can be any arbitrary string, and is used to denote
        annotator responses in `measure_quality`.
        """
        pass

    @abstractmethod
    def measure_quality(self, item: TestItemAnnotations) -> Dict[str, float]:
        """Use the SUT responses with annotations to determine how well the SUT did on this TestItem."""
        pass

    @abstractmethod
    def aggregate_measurements(self, items: List[MeasuredTestItem]) -> Typeable:
        """Combine the measurements for each TestItem into a test specific Typeable."""
        pass


class TestResult(TypedData):
    """Container for plugin defined Test result data.

    Every Test can return data however it wants, so this generically
    records the Test's return type and data.
    You can use `to_instance` to get back to the original form.
    """

    # Convince pytest to ignore this class.
    __test__ = False
