from enum import Enum
from pydantic import BaseModel
from typing import List, Optional


class SUTOptions(BaseModel):
    """
    An exhaustive set of options that could potentially be desired by a SUT.

    Not all SUTs respect all options.
    """

    num_completions: int = 1
    """Generate this many completions (by sampling from the model)"""

    max_tokens: int = 100
    """Maximum number of tokens to generate (per completion)"""

    temperature: Optional[float] = None
    """Temperature parameter that governs diversity"""

    top_k_per_token: Optional[int] = None
    """Take this many highest probability candidates per token in the completion"""

    stop_sequences: Optional[List[str]] = None
    """Stop generating once we hit one of these strings."""

    top_p: Optional[float] = None
    """Same from tokens that occupy this probability mass (nucleus sampling)"""

    presence_penalty: Optional[float] = None
    """Penalize repetition (OpenAI & Writer only)"""

    frequency_penalty: Optional[float] = None
    """Penalize repetition (OpenAI & Writer only)"""

    random: Optional[str] = None
    """Used to control randomness. Expect different responses for the same
    request but with different values for `random`."""

    # Must specify SUTCapabilities for these
    top_logprobs: Optional[int] = None
    """If present, will request the log probabilities for this
    many of the top tokens at each token position."""


class ChatRole(str, Enum):
    user = "USER"
    sut = "SUT"
    system = "SYSTEM"


class ChatMessage(BaseModel):
    text: str
    role: ChatRole


class ChatPrompt(BaseModel):
    messages: List[ChatMessage]
    options: SUTOptions = SUTOptions()


class TextPrompt(BaseModel, frozen=True):
    """What actually goes to the SUT."""

    text: str
    options: SUTOptions = SUTOptions()
