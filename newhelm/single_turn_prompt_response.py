from typing import Dict, List, Mapping, Type, TypeVar

from pydantic import BaseModel, Field
from newhelm.annotation import Annotation

from newhelm.placeholders import Prompt
from newhelm.sut import SUTResponse
from newhelm.typed_data import TypedData

# TODO: This whole file assumes single turn. We'll either need to make it
# more complicated, or make parallel structures for multi-turn.

_BaseModelType = TypeVar("_BaseModelType", bound=BaseModel)
_Context = TypedData | str | Mapping | None


def resolve_context_type(context: _Context, cls):
    if issubclass(cls, BaseModel):
        assert isinstance(context, TypedData)
        return context.to_instance(cls)
    if isinstance(cls, str):
        return context
    if isinstance(cls, Mapping):
        return context
    raise AssertionError("Unhandled context type:", cls)


class PromptWithContext(BaseModel):
    """Combine a prompt with arbitrary context data."""

    prompt: Prompt
    """The data that goes to the SUT."""

    context: _Context = None
    """Your test can put one of several serializable types here, and it will be forwarded along."""

    def get_context(self, cls):
        """Convenience function for strongly typing the context."""
        return resolve_context_type(self.context, cls)


class TestItem(BaseModel):
    """This is the smallest unit in a Test that can be judged for quality.

    For many Tests, this will be a single Prompt.
    """

    prompts: List[PromptWithContext]

    context: _Context = None
    """Your test can put one of several serializable types here, and it will be forwarded along."""

    def get_context(self, cls):
        """Convenience function for strongly typing the context."""
        return resolve_context_type(self.context, cls)

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

    def get_annotation(self, key: str, cls: Type[_BaseModelType]) -> _BaseModelType:
        """Convenience function for getting strongly typed annotations."""
        annotation = self.annotations[key]
        return annotation.to_instance(cls)

    # Convince pytest to ignore this class.
    __test__ = False


class MeasuredTestItem(BaseModel):
    """A TestItem with its measurement of quality.

    Note, this does NOT include any SUT Responses or Annotations, as that should already be baked into the Measurements.
    """

    test_item: TestItem
    measurements: Dict[str, float]
