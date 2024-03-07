from typing import Dict, List, Mapping, Optional, Type, TypeVar

from pydantic import BaseModel, Field
from newhelm.annotation import Annotation

from newhelm.prompt import ChatPrompt, TextPrompt
from newhelm.sut import SUTResponse
from newhelm.typed_data import TypedData

# TODO: This whole file assumes single turn. We'll either need to make it
# more complicated, or make parallel structures for multi-turn.

_BaseModelType = TypeVar("_BaseModelType", bound=BaseModel)
_Context = TypedData | str | Mapping | None


class PromptWithContext(BaseModel):
    """Combine a prompt with arbitrary context data."""

    prompt: TextPrompt | ChatPrompt
    """The data that goes to the SUT."""

    source_id: Optional[str] = None
    """Identifier for where this Prompt came from in the underlying datasource."""

    @property
    def context(self):
        """Your test can add one of several serializable types as context, and it will be forwarded."""
        if isinstance(self.context_internal, TypedData):
            return self.context_internal.to_instance()
        return self.context_internal

    context_internal: _Context = None
    """Internal variable for the serialization friendly version of context"""

    def __init__(self, *, prompt, source_id=None, context=None, context_internal=None):
        if context_internal is not None:
            internal = context_internal
        elif isinstance(context, BaseModel):
            internal = TypedData.from_instance(context)
        else:
            internal = context
        super().__init__(prompt=prompt, source_id=source_id, context_internal=internal)


class TestItem(BaseModel):
    """This is the smallest unit in a Test that can be judged for quality.

    For many Tests, this will be a single Prompt.
    """

    prompts: List[PromptWithContext]

    @property
    def context(self):
        """Your test can add one of several serializable types as context, and it will be forwarded."""
        if isinstance(self.context_internal, TypedData):
            return self.context_internal.to_instance()
        return self.context_internal

    context_internal: _Context = None
    """Internal variable for the serialization friendly version of context"""

    def __init__(self, *, prompts, context=None, context_internal=None):
        if context_internal is not None:
            internal = context_internal
        elif isinstance(context, BaseModel):
            internal = TypedData.from_instance(context)
        else:
            internal = context
        super().__init__(prompts=prompts, context_internal=internal)

    # Convince pytest to ignore this class.
    __test__ = False


class PromptInteraction(BaseModel):
    """Combine a Prompt with the SUT Response to make it easier for Tests to measure quality."""

    prompt: PromptWithContext
    response: SUTResponse


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
