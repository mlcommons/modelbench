from pydantic import BaseModel

from typing import Any, Dict, List, Optional


class SafetyAnnotation(BaseModel):
    """Annotation object produced by all annotators."""

    is_safe: bool
    is_valid: bool = True

    metadata: Optional[dict] = None


class EnsembleSafetyAnnotation(SafetyAnnotation):
    """Dataclass for safety model responses computed by an ensemble

    Stores additional metadata on the original responses

    **joined_responses**: Store the individual SafetyModelResponse objects
    """

    joined_responses: Dict[str, SafetyAnnotation]
