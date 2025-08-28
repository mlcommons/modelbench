from modelgauge.annotator import CompletionAnnotator
from modelgauge.single_turn_prompt_response import TestItem
from modelgauge.sut import SUTResponse
from pydantic import BaseModel


class FakeAnnotation(BaseModel):
    sut_text: str


class FakeAnnotatorRequest(BaseModel):
    text: str


class FakeAnnotatorResponse(BaseModel):
    sut_text: str


class FakeAnnotator(CompletionAnnotator[FakeAnnotation]):
    """Fake annotator that just returns the first completion from the SUT."""

    def __init__(self, uid):
        super().__init__(uid)
        self.annotate_calls = 0

    def translate_request(self, test_item: TestItem, response: SUTResponse):
        return FakeAnnotatorRequest(text=response.text)

    def annotate(self, annotation_request: FakeAnnotatorRequest):
        """Returns an annotation for a single TestItem's interactions."""
        self.annotate_calls += 1
        return FakeAnnotatorResponse(sut_text=annotation_request.text)

    def translate_response(self, request, response: FakeAnnotatorResponse) -> FakeAnnotation:
        return FakeAnnotation(sut_text=response.sut_text)


class BadAnnotator(FakeAnnotator):
    """Fake annotator that always fails to annotate."""

    def annotate(self, annotation_request: FakeAnnotatorRequest):
        """Always raises an error when trying to annotate."""
        raise RuntimeError("Failed to annotate")
