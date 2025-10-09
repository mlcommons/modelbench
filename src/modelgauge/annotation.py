from pydantic import BaseModel

from typing import Any, Dict, List, Optional


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
