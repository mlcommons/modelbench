from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Callable, List, Optional


@dataclass(frozen=True, kw_only=True)
class SUTOptions:
    """
    An exhaustive set of options that could potentially be desired by a SUT.

    Not all SUTs respect all options.
    """

    temperature: float = 1.0
    """Temperature parameter that governs diversity"""

    num_completions: int = 1
    """Generate this many completions (by sampling from the model)"""

    top_k_per_token: int = 1
    """Take this many highest probability candidates per token in the completion"""

    max_tokens: int = 100
    """Maximum number of tokens to generate (per completion)"""

    stop_sequences: List[str] = field(default_factory=list)
    """Stop generating once we hit one of these strings."""

    echo_prompt: bool = False
    """Should `prompt` be included as a prefix of each completion? (e.g., for
    evaluating perplexity of the prompt)"""

    top_p: float = 1
    """Same from tokens that occupy this probability mass (nucleus sampling)"""

    presence_penalty: float = 0
    """Penalize repetition (OpenAI & Writer only)"""

    frequency_penalty: float = 0
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
