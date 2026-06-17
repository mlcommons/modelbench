from abc import abstractmethod
from typing import Optional, Sequence, Type

from pydantic import BaseModel

from modelgauge.model_options import ModelOptions, TopTokens
from modelgauge.not_implemented import not_implemented
from modelgauge.prompt import ChatPrompt, TextPrompt
from modelgauge.ready import Readyable, ReadyResponse
from modelgauge.record_init import InitializationRecord
from modelgauge.sut_capabilities import SUTCapability
from modelgauge.tracked_object import TrackedObject

REFUSAL_RESPONSE = ""


class SUTResponse(BaseModel):
    """The data that came out of the SUT."""

    text: str
    reasoning: Optional[str] = None
    top_logprobs: Optional[Sequence[TopTokens]] = None
    """For each position, list the probabilities for each of the most likely tokens.

    To guarantee this field is not None, the Test must specify ModelOptions.top_logprobs
    and that it requires_sut_capabilities ProducesPerTokenLogProbabilities.
    SUTs that set this value must specify they have the ProducesPerTokenLogProbabilities
    capability. They may conditional setting the field on on ModelOptions.top_logprobs being not None.
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
_READINESS_CHECK_SUT_OPTIONS = ModelOptions(max_tokens=20)


class PromptResponseSUT(SUT, Readyable):
    """
    Abstract base class that provides an interface to any SUT that is designed for handling a single-turn.
    """

    def __init__(self, uid: str):
        super().__init__(uid)
        self.reasoning_handler: Optional[Type[ReasoningHandler]] = ReasoningHandler.sut_matches(self)

    def run_readiness_check(self) -> ReadyResponse:
        raw_request = self.translate_text_prompt(_READINESS_CHECK_TEXT_PROMPT, options=_READINESS_CHECK_SUT_OPTIONS)
        raw_response = self.evaluate(raw_request)
        response = self.translate_response(raw_request, raw_response)
        return ReadyResponse(is_ready=response.text is not None, response=response)

    def translate_text_prompt(self, prompt: TextPrompt, options: ModelOptions):
        if self.reasoning_handler is not None:
            return self.reasoning_handler.translate_text_prompt(self, prompt, options)
        return self._translate_text_prompt(prompt, options)

    @not_implemented
    def _translate_text_prompt(self, prompt: TextPrompt, options: ModelOptions):
        """Convert the prompt + SUT options into the SUT's native representation.

        This method must be implemented if the SUT accepts text prompts.
        """
        raise NotImplementedError(f"SUT {self.__class__.__name__} does not implement translate_text_prompt.")

    @not_implemented
    def translate_chat_prompt(self, prompt: ChatPrompt, options: ModelOptions):
        """Convert the prompt + SUT options into the SUT's native representation.

        This method must be implemented if the SUT accepts chat prompts.
        """
        raise NotImplementedError(f"SUT {self.__class__.__name__} does not implement translate_chat_prompt.")

    def evaluate(self, request):
        """Evaluate this SUT on the native request."""
        if self.reasoning_handler is not None:
            return self.reasoning_handler.evaluate(self, request)
        return self._evaluate(request)

    @abstractmethod
    def _evaluate(self, request):
        """Evaluate this SUT on the native request."""
        pass

    def translate_response(self, request, response) -> SUTResponse:
        """Convert the native response into a form all Tests can process."""
        if self.reasoning_handler is not None:
            return self.reasoning_handler._translate_response(self, request, response)
        return self._translate_response(request, response)

    @abstractmethod
    def _translate_response(self, request, response) -> SUTResponse:
        """Convert the native response into a form all Tests can process."""
        pass
