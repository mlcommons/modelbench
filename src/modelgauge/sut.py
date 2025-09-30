from abc import abstractmethod
from typing import List, Optional, Sequence, Type

from pydantic import BaseModel, model_validator

from modelgauge.not_implemented import not_implemented
from modelgauge.prompt import ChatPrompt, TextPrompt
from modelgauge.ready import Readyable, ReadyResponse
from modelgauge.record_init import InitializationRecord
from modelgauge.sut_capabilities import SUTCapability
from modelgauge.tracked_object import TrackedObject

REFUSAL_RESPONSE = ""


class SUTOptions(BaseModel):
    """
    An exhaustive set of options that could potentially be desired by a SUT.

    Not all SUTs respect all options.
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
                f"Invalid SUTOptions. max_total_output_tokens ({self.max_total_output_tokens}) must be >= max_tokens ({self.max_tokens})."
            )
        return self

    @staticmethod
    def create_from_arguments(max_tokens=None, temp=None, top_p=None, top_k=None, top_logprobs=None):
        options = SUTOptions()
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


class TokenProbability(BaseModel):
    """Probability assigned to a given token."""

    token: str
    logprob: float


class TopTokens(BaseModel):
    """List of most likely tokens and their probabilities."""

    top_tokens: Sequence[TokenProbability]


class SUTResponse(BaseModel):
    """The data that came out of the SUT."""

    text: str
    top_logprobs: Optional[Sequence[TopTokens]] = None
    """For each position, list the probabilities for each of the most likely tokens.

    To guarantee this field is not None, the Test must specify SUTOptions.top_logprobs
    and that it requires_sut_capabilities ProducesPerTokenLogProbabilities.
    SUTs that set this value must specify they have the ProducesPerTokenLogProbabilities
    capability. They may conditional setting the field on on SUTOptions.top_logprobs being not None.
    """


class SUT(TrackedObject):
    """Base class for all SUTs.

    SUT capabilities can be specified with the `@modelgauge_sut` decorator.
    There is no guaranteed interface between SUTs, so no methods here.

    Attributes:
        uid (str): Unique identifier for this SUT.
        capabilities: List of capabilities this SUT has.
        initialization_record: The record of args and kwargs the SUT was initialized with.
    """

    # Set automatically by @modelgauge_sut()
    capabilities: Sequence[Type[SUTCapability]]

    def __init__(self, uid: str):
        super().__init__(uid)
        # The initialization record is set automatically by @modelgauge_sut()
        self.initialization_record: InitializationRecord


_READINESS_CHECK_TEXT_PROMPT = TextPrompt(text="Why did the chicken cross the road?")
_READINESS_CHECK_SUT_OPTIONS = SUTOptions(max_tokens=20)


class PromptResponseSUT(SUT, Readyable):
    """
    Abstract base class that provides an interface to any SUT that is designed for handling a single-turn.
    """

    def run_readiness_check(self) -> ReadyResponse:
        raw_request = self.translate_text_prompt(_READINESS_CHECK_TEXT_PROMPT, options=_READINESS_CHECK_SUT_OPTIONS)
        raw_response = self.evaluate(raw_request)
        response = self.translate_response(raw_request, raw_response)
        return ReadyResponse(is_ready=response.text is not None, response=response)

    @not_implemented
    def translate_text_prompt(self, prompt: TextPrompt, options: SUTOptions):
        """Convert the prompt + SUT options into the SUT's native representation.

        This method must be implemented if the SUT accepts text prompts.
        """
        raise NotImplementedError(f"SUT {self.__class__.__name__} does not implement translate_text_prompt.")

    @not_implemented
    def translate_chat_prompt(self, prompt: ChatPrompt, options: SUTOptions):
        """Convert the prompt + SUT options into the SUT's native representation.

        This method must be implemented if the SUT accepts chat prompts.
        """
        raise NotImplementedError(f"SUT {self.__class__.__name__} does not implement translate_chat_prompt.")

    @abstractmethod
    def evaluate(self, request):
        """Evaluate this SUT on the native request."""
        pass

    @abstractmethod
    def translate_response(self, request, response) -> SUTResponse:
        """Convert the native response into a form all Tests can process."""
        pass
