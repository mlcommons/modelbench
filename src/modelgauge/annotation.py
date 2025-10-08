from pydantic import BaseModel

from typing import Any, Dict, List, Optional


from modelgauge.typed_data import TypedData


# TODO: Delete
class Annotation(TypedData):
    """Container for plugin defined annotation data.

    Every annotator can return data however it wants.
    Since Tests are responsible for both deciding what
    Annotators to apply and how to interpret their results,
    they can use `to_instance` to get it back in the form they want.
    """

    pass


class SafetyAnnotation(BaseModel):
    """Annotation object produced by all annotators."""

    is_safe: bool
    is_valid: bool = True

    metadata: Optional[dict] = None


# TODO: Delete!
class SafetyModelResponse(SafetyAnnotation):
    """Dataclass that ensemble component responses should adhere to"""

    safety_categories: List[str]
    is_valid: bool = True
    raw_input: Optional[str] = None
    raw_response: Optional[str] = None
    reasoning: Optional[str] = None
    total_tokens_used: Optional[float] = None
    logprobs: Optional[Any] = None


# TODO: Rename-- EnsembleSafetyAnnotation
class EnsembleSafetyModelResponse(SafetyAnnotation):
    """Dataclass for safety model responses computed by an ensemble

    Stores additional metadata on the original responses

    **joined_responses**: Store the individual SafetyModelResponse objects
    """

    joined_responses: Dict[str, SafetyAnnotation]
