from typing import Dict, Optional, Sequence

from pydantic import BaseModel

from modelgauge.model_options import TopTokens


class MissingAnnotationError(Exception):
    pass


class InvalidAnnotationError(Exception):
    pass


class SafetyAnnotation(BaseModel):
    """Annotation object produced by all annotators."""

    is_safe: bool
    is_valid: bool = True
    top_logprobs: Optional[Sequence[TopTokens]] = None

    metadata: Optional[dict] = None


class EnsembleSafetyAnnotation(SafetyAnnotation):
    """Dataclass for safety annotation computed by an ensemble

    Stores additional metadata on the original responses
    """

    joined_responses: Dict[str, SafetyAnnotation]
