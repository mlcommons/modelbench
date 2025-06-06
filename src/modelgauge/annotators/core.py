from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class SafetyModelResponse(BaseModel):
    """Dataclass that all safety model responses should adhere to"""

    is_safe: bool
    safety_categories: List[str]
    is_valid: bool = True
    raw_input: Optional[str] = None
    raw_response: Optional[str] = None
    reasoning: Optional[str] = None
    total_tokens_used: Optional[float] = None
    logprobs: Optional[Any] = None


class EnsembleSafetyModelResponse(SafetyModelResponse):
    """Dataclass for safety model responses computed by an ensemble

    Stores additional metadata on the original responses

    **joined_responses**: Store the individual SafetyModelResponse objects
    """

    joined_responses: Dict[str, SafetyModelResponse]
