from abc import abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

from modelgauge.prompt import ChatPrompt, TextPrompt
from modelgauge.ready import Readyable, ReadyResponse
from modelgauge.single_turn_prompt_response import TestItem
from modelgauge.sut import _READINESS_CHECK_TEXT_PROMPT, SUTResponse
from modelgauge.tracked_object import TrackedObject

AnnotationType = TypeVar("AnnotationType", bound=BaseModel)
_READINESS_CHECK_SOURCE_ID = "ignored"
_READINESS_CHECK_TEST_ITEM = TestItem(
    prompt=_READINESS_CHECK_TEXT_PROMPT,
    source_id=_READINESS_CHECK_SOURCE_ID,
)
_READINESS_CHECK_SUT_RESPONSE = SUTResponse(text="To get to the other side.")


class Annotator(TrackedObject):
    """The base class for all annotators."""

    def __init__(self, uid):
        super().__init__(uid)


class CompletionAnnotator(Annotator, Generic[AnnotationType], Readyable):
    """Annotator that examines a single prompt+completion pair at a time.

    Subclasses can report whatever class they want, as long as it inherits from Pydantic's BaseModel.
    """

    def run_readiness_check(self) -> ReadyResponse:
        raw_request = self.translate_request(_READINESS_CHECK_TEST_ITEM, _READINESS_CHECK_SUT_RESPONSE)
        raw_response = self.annotate(raw_request)
        response = self.translate_response(raw_request, raw_response)
        return ReadyResponse(is_ready=bool(response), response=response)

    def translate_request(self, test_item: TestItem, response: SUTResponse):
        return self.translate_prompt(test_item.evaluated_prompt, response)

    @abstractmethod
    def translate_prompt(self, prompt: TextPrompt | ChatPrompt, response: SUTResponse):
        """Convert the prompt+response into the native representation for this annotator."""
        pass

    @abstractmethod
    def annotate(self, annotation_request):
        """Perform annotation and return the raw response from the annotator."""
        pass

    @abstractmethod
    def translate_response(self, request, response) -> AnnotationType:
        """Convert the raw response into the form read by Tests."""
        pass
