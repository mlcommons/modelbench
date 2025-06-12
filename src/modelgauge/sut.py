from abc import ABC, abstractmethod
from typing import Generic, List, Optional, Sequence, Type, TypeVar

from modelgauge.not_implemented import not_implemented
from modelgauge.prompt import ChatPrompt, TextPrompt
from modelgauge.record_init import InitializationRecord
from modelgauge.sut_capabilities import SUTCapability
from modelgauge.tracked_object import TrackedObject
from pydantic import BaseModel

RequestType = TypeVar("RequestType")
ResponseType = TypeVar("ResponseType")

REFUSAL_RESPONSE = ""


class SUTNotFoundException(Exception):
    pass


class SUTOptions(BaseModel):
    """
    An exhaustive set of options that could potentially be desired by a SUT.

    Not all SUTs respect all options.
    """

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


class PromptResponseSUT(SUT, ABC, Generic[RequestType, ResponseType]):
    """
    Abstract base class that provides an interface to any SUT that is designed for handling a single-turn.

    This class uses generics to allow for any type of native request and response objects.
    """

    @not_implemented
    def translate_text_prompt(self, prompt: TextPrompt, options: SUTOptions) -> RequestType:
        """Convert the prompt + SUT options into the SUT's native representation.

        This method must be implemented if the SUT accepts text prompts.
        """
        raise NotImplementedError(f"SUT {self.__class__.__name__} does not implement translate_text_prompt.")

    @not_implemented
    def translate_chat_prompt(self, prompt: ChatPrompt, options: SUTOptions) -> RequestType:
        """Convert the prompt + SUT options into the SUT's native representation.

        This method must be implemented if the SUT accepts chat prompts.
        """
        raise NotImplementedError(f"SUT {self.__class__.__name__} does not implement translate_chat_prompt.")

    @abstractmethod
    def evaluate(self, request: RequestType) -> ResponseType:
        """Evaluate this SUT on the native request."""
        pass

    @abstractmethod
    def translate_response(self, request: RequestType, response: ResponseType) -> SUTResponse:
        """Convert the native response into a form all Tests can process."""
        pass
