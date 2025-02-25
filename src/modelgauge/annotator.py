from abc import ABC, abstractmethod
from modelgauge.single_turn_prompt_response import TestItem
from modelgauge.sut import SUTResponse
from modelgauge.tracked_object import TrackedObject
from pydantic import BaseModel
from typing import Generic, TypeVar

AnnotationType = TypeVar("AnnotationType", bound=BaseModel)


class Annotator(TrackedObject):
    """The base class for all annotators."""

    def __init__(self, uid):
        super().__init__(uid)


class CompletionAnnotator(Annotator, Generic[AnnotationType]):
    """Annotator that examines a single prompt+completion pair at a time.

    Subclasses can report whatever class they want, as long as it inherits from Pydantic's BaseModel.
    """

    @abstractmethod
    def translate_request(self, test_item: TestItem, response: SUTResponse):
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
