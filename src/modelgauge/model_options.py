from typing import Optional, List

from pydantic import BaseModel, model_validator


class ModelOptions(BaseModel):
    """
    An exhaustive set of options that could potentially be desired by a model.

    Not all SUTs and annotators respect all options.
    """

    max_tokens: int = 100
    """Maximum number of tokens to generate (per completion)"""

    max_total_output_tokens: Optional[int] = None
    """Maximum number of tokens for all generated SUT outputs, including reasoning."""

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

    @model_validator(mode="after")
    def check_max_total_output_tokens(self):
        if self.max_total_output_tokens is not None and self.max_total_output_tokens < self.max_tokens:
            raise ValueError(
                f"Invalid ModelOptions. max_total_output_tokens ({self.max_total_output_tokens}) must be >= max_tokens ({self.max_tokens})."
            )
        return self

    @staticmethod
    def create_from_arguments(max_tokens=None, temp=None, top_p=None, top_k=None, top_logprobs=None):
        options = ModelOptions()
        if max_tokens is not None:
            options.max_tokens = max_tokens
        if temp is not None:
            options.temperature = temp
        if top_p is not None:
            options.top_p = top_p
        if top_k is not None:
            options.top_k_per_token = top_k
        if top_logprobs is not None:
            options.top_logprobs = top_logprobs

        return options
