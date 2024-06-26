from abc import ABC, abstractmethod
from modelgauge.not_implemented import not_implemented
from modelgauge.prompt import ChatPrompt, TextPrompt
from modelgauge.record_init import InitializationRecord
from modelgauge.sut_capabilities import SUTCapability
from modelgauge.tracked_object import TrackedObject
from pydantic import BaseModel
from typing import Generic, List, Optional, Sequence, Type, TypeVar

RequestType = TypeVar("RequestType")
ResponseType = TypeVar("ResponseType")


class TokenProbability(BaseModel):
    """Probability assigned to a given token."""

    token: str
    logprob: float


class TopTokens(BaseModel):
    """List of most likely tokens and their probabilities."""

    top_tokens: Sequence[TokenProbability]


class SUTCompletion(BaseModel):
    """All data about a single completion in the response."""

    text: str
    top_logprobs: Optional[Sequence[TopTokens]] = None
    """For each position, list the probabilities for each of the most likely tokens.

    To guarantee this field is not None, the Test must specify SUTOptions.top_logprobs
    and that it requires_sut_capabilities ProducesPerTokenLogProbabilities.
    SUTs that set this value must specify they have the ProducesPerTokenLogProbabilities
    capability. They may conditional setting the field on on SUTOptions.top_logprobs being not None.
    """


class SUTResponse(BaseModel):
    """The data that came out of the SUT."""

    completions: List[SUTCompletion]


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
    def translate_text_prompt(self, prompt: TextPrompt) -> RequestType:
        """Convert the prompt into the SUT's native representation.

        This method must be implemented if the SUT accepts text prompts.
        """
        raise NotImplementedError(
            f"SUT {self.__class__.__name__} does not implement translate_text_prompt."
        )

    @not_implemented
    def translate_chat_prompt(self, prompt: ChatPrompt) -> RequestType:
        """Convert the prompt into the SUT's native representation.

        This method must be implemented if the SUT accepts chat prompts.
        """
        raise NotImplementedError(
            f"SUT {self.__class__.__name__} does not implement translate_chat_prompt."
        )

    @abstractmethod
    def evaluate(self, request: RequestType) -> ResponseType:
        """Evaluate this SUT on the native request."""
        pass

    @abstractmethod
    def translate_response(
        self, request: RequestType, response: ResponseType
    ) -> SUTResponse:
        """Convert the native response into a form all Tests can process."""
        pass
