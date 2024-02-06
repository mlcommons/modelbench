from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Callable, List, Optional


@dataclass(frozen=True, kw_only=True)
class SUTOptions:
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

    echo_prompt: Optional[bool] = None
    # TODO Remove this.

    top_p: Optional[float] = None
    """Same from tokens that occupy this probability mass (nucleus sampling)"""

    presence_penalty: Optional[float] = None
    """Penalize repetition (OpenAI & Writer only)"""

    frequency_penalty: Optional[float] = None
    """Penalize repetition (OpenAI & Writer only)"""

    random: Optional[str] = None
    """Used to control randomness. Expect different responses for the same
    request but with different values for `random`."""


@dataclass(frozen=True)
class Prompt:
    """What actually goes to the SUT."""

    text: str
    options: SUTOptions = SUTOptions()


@dataclass(frozen=True)
class Result:
    """The measurement produced by Test."""

    # Just a placeholder.
    name: str
    value: float
