from typing import Dict, Mapping, Optional, Type, TypeVar
from dataclasses import dataclass

from pydantic import BaseModel, Field, model_validator

from modelgauge.annotation import Annotation, SafetyAnnotation
from modelgauge.prompt import ChatPrompt, TextPrompt
from modelgauge.sut import SUTResponse
from modelgauge.typed_data import TypedData

# TODO: This whole file assumes single turn. We'll either need to make it
# more complicated, or make parallel structures for multi-turn.

_BaseModelType = TypeVar("_BaseModelType", bound=BaseModel)
_Context = TypedData | str | Mapping | None


class TestItem(BaseModel):
    """Combine a prompt with arbitrary context data.
    This is the smallest unit in a Test that can be judged for quality."""

    prompt: TextPrompt | ChatPrompt
    """The data that goes to the SUT."""

    evaluated_prompt: TextPrompt | ChatPrompt
    """The data that goes to the annotator. Defaults to `prompt`."""

    source_id: Optional[str]
    """Identifier for where this Prompt came from in the underlying datasource."""

    @model_validator(mode="before")
    def set_default_evaluated_prompt(cls, values: dict):
        """Set evaluated_prompt to regular prompt if not provided."""
        if "evaluated_prompt" not in values or values["evaluated_prompt"] is None:
            values["evaluated_prompt"] = values.get("prompt")
        return values

    @property
    def context(self):
        """Your test can add one of several serializable types as context, and it will be forwarded."""
        if isinstance(self.context_internal, TypedData):
            return self.context_internal.to_instance()
        return self.context_internal

    context_internal: _Context = None
    """Internal variable for the serialization friendly version of context"""

    def __hash__(self):
        if self.source_id:
            return hash(self.source_id) + hash(self.prompt.text)
        else:
            return hash(self.prompt.text)

    def __init__(self, *, prompt, source_id, context=None, context_internal=None, **kwargs):
        if context_internal is not None:
            internal = TypedData.model_validate(context_internal)
        elif isinstance(context, BaseModel):
            internal = TypedData.from_instance(context)
        else:
            internal = context
        super().__init__(prompt=prompt, source_id=source_id, context_internal=internal, **kwargs)


class SUTResponseAnnotations(BaseModel):
    """The annotations for a SUT Response to a single TestItem."""

    test_item: TestItem
    sut_response: SUTResponse
    annotations: Dict[str, SafetyAnnotation] = Field(default_factory=dict)
    """All of the annotations, keyed by annotator."""

    def get_annotation(self, key: str) -> SafetyAnnotation:
        return self.annotations[key]


class MeasuredTestItem(BaseModel):
    """A TestItem with its measurement of quality.

    Note, this does NOT include any SUT Responses or Annotations, as that should already be baked into the Measurements.
    """

    test_item: TestItem
    measurements: Dict[str, float]


@dataclass
class SUTInteraction:
    prompt: TestItem
    sut_uid: str
    response: SUTResponse

    def __hash__(self):
        return hash(self.prompt.source_id + self.sut_uid)


@dataclass
class AnnotatedSUTInteraction:
    annotator_uid: str
    annotation: Annotation
    sut_interaction: SUTInteraction

    def __hash__(self):
        return hash(self.prompt.source_id + self.sut_uid + self.annotator_uid)
