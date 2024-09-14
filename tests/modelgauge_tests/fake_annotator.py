from modelgauge.annotator import CompletionAnnotator
from modelgauge.single_turn_prompt_response import PromptWithContext
from modelgauge.sut import SUTCompletion
from pydantic import BaseModel


class FakeAnnotation(BaseModel):
    sut_text: str


class FakeAnnotatorRequest(BaseModel):
    text: str


class FakeAnnotatorResponse(BaseModel):
    sut_text: str


class FakeAnnotator(CompletionAnnotator[FakeAnnotation]):
    """Fake annotator that just returns the first completion from the SUT."""

    def __init__(self):
        self.annotate_calls = 0

    def translate_request(self, prompt: PromptWithContext, completion: SUTCompletion):
        return FakeAnnotatorRequest(text=completion.text)

    def annotate(self, annotation_request: FakeAnnotatorRequest):
        """Returns an annotation for a single TestItem's interactions."""
        self.annotate_calls += 1
        return FakeAnnotatorResponse(sut_text=annotation_request.text)

    def translate_response(
        self, request, response: FakeAnnotatorResponse
    ) -> FakeAnnotation:
        return FakeAnnotation(sut_text=response.sut_text)
