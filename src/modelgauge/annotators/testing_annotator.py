from modelgauge.annotator import CompletionAnnotator
from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.single_turn_prompt_response import TestItem
from modelgauge.sut import SUTResponse
from pydantic import BaseModel


class TestingAnnotation(BaseModel):
    """An annotation returning is_safe for use by unit tests only."""

    is_safe: float


class TestingRequest(BaseModel):
    text: str


class TestingResponse(BaseModel):
    score: float


class TestingAnnotator(CompletionAnnotator[TestingAnnotation]):
    """An annotator used for unit tests only"""

    def translate_request(self, test_item: TestItem, response: SUTResponse):
        return TestingRequest(text=response.text)

    def annotate(self, annotation_request: TestingRequest) -> TestingResponse:
        return TestingResponse(score=0.0)

    def translate_response(self, request, response: TestingResponse) -> TestingAnnotation:
        return TestingAnnotation(is_safe=0.0)


ANNOTATORS.register(TestingAnnotator, "testing_annotator")
