from modelgauge.annotator import CompletionAnnotator
from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.single_turn_prompt_response import TestItem
from modelgauge.sut import SUTResponse
from pydantic import BaseModel


class FakeAnnotation(BaseModel):
    """An annotation returning is_safe for use by unit tests only."""

    is_safe: float


class FakeRequest(BaseModel):
    text: str


class FakeResponse(BaseModel):
    score: float


class FakeAnnotator(CompletionAnnotator[FakeAnnotation]):
    """An annotator used for unit tests only"""

    def translate_request(self, test_item: TestItem, response: SUTResponse):
        return FakeRequest(text=response.text)

    def annotate(self, annotation_request: FakeRequest) -> FakeResponse:
        return FakeResponse(score=0.0)

    def translate_response(self, request, response: FakeResponse) -> FakeAnnotation:
        return FakeAnnotation(is_safe=0.0)


ANNOTATORS.register(FakeAnnotator, "fake_annotator")
