from typing import Any, Dict, List, Type, TypeVar

from pydantic import BaseModel, Field
from newhelm.annotation import Annotation

from newhelm.placeholders import Prompt
from newhelm.sut import SUTResponse

# TODO: This whole file assumes single turn. We'll either need to make it
# more complicated, or make parallel structures for multi-turn.

_InT = TypeVar("_InT")


class PromptWithContext(BaseModel):
    """Combine a prompt with arbitrary context data."""

    prompt: Prompt
    """The data that goes to the SUT."""

    context: Any = None
    """Your test can put anything that can be serialized here, and it will be forwarded along."""


class TestItem(BaseModel):
    """This is the smallest unit in a Test that can be judged for quality.

    For many Tests, this will be a single Prompt.
    """

    prompts: List[PromptWithContext]

    context: Any = None
    """Your test can put anything that can be serialized here, and it will be forwarded along."""

    # Convince pytest to ignore this class.
    __test__ = False


class PromptInteraction(BaseModel):
    """Combine a Prompt with the SUT Response to make it easier for Tests to measure quality."""

    prompt: PromptWithContext
    response: SUTResponse


class TestItemInteractions(BaseModel):
    """All of the Interactions with a SUT for a single TestItem."""

    interactions: List[PromptInteraction]

    # TODO: This duplicates the list of prompts in the object.
    # Maybe denormalize here.
    test_item: TestItem


class TestItemAnnotations(BaseModel):
    """All of the Interactions with a SUT plus their annotations for a single TestItem."""

    # TODO: This duplicates the list of prompts in the object.
    # Maybe denormalize here.
    test_item: TestItem

    interactions: List[PromptInteraction]

    annotations: Dict[str, Annotation] = Field(default_factory=dict)
    """All of the annotations, keyed by annotator.
    
    
    Note: This asserts that annotations are at the TestItem level and not
    associated with individual Prompts. This is in keeping with the idea that
    if two Prompts can be measured separately, they should be separate TestItems.
    """

    def get_annotation(self, key: str, cls: Type[_InT]) -> _InT:
        """Convenience function for getting strongly typed annotations."""
        annotation = self.annotations[key]
        assert isinstance(annotation, cls)
        return annotation

    # Convince pytest to ignore this class.
    __test__ = False


class MeasuredTestItem(BaseModel):
    """A TestItem with its measurement of quality.

    Note, this does NOT include any SUT Responses or Annotations, as that should already be baked into the Measurements.
    """

    test_item: TestItem
    measurements: Dict[str, float]
