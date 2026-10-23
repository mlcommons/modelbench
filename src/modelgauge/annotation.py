from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence

from pydantic import BaseModel

from modelgauge.model_options import TopTokens


class MissingAnnotationError(Exception):
    pass


class InvalidAnnotationError(Exception):
    pass


class Annotation(BaseModel, ABC):
    @abstractmethod
    def dump_value(self) -> dict[str, Any]:
        """Dump the main annotation value into a dict."""
        pass


class SafetyAnnotation(Annotation):
    is_safe: bool
    is_valid: bool = True
    top_logprobs: Optional[Sequence[TopTokens]] = None

    metadata: Optional[dict] = None

    def dump_value(self) -> dict[str, Any]:
        return {
            "is_safe": self.is_safe,
        }


class ViolationProbabilityAnnotation(Annotation):
    violation_probability: float | None
    is_valid: bool
    metadata: dict[str, Any] | None = None

    def dump_value(self) -> dict[str, Any]:
        return {
            "violation_probability": self.violation_probability,
        }


class EnsembleSafetyAnnotation(SafetyAnnotation):
    """Dataclass for safety annotation computed by an ensemble

    Stores additional metadata on the original responses
    """

    joined_responses: Dict[str, SafetyAnnotation]
