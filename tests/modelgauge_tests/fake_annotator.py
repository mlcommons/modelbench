from modelgauge.annotation import SafetyAnnotation
from modelgauge.annotator import Annotator
from modelgauge.sut import SUTResponse
from pydantic import BaseModel


class FakeAnnotatorRequest(BaseModel):
    text: str


class FakeAnnotatorResponse(BaseModel):
    sut_text: str


class FakeSafetyAnnotator(Annotator):

    def __init__(self, uid):
        super().__init__(uid)
        self.annotate_calls = 0

    def translate_prompt(self, prompt, response: SUTResponse):
        return FakeAnnotatorRequest(text=response.text)

    def annotate(self, annotation_request: FakeAnnotatorRequest):
        self.annotate_calls += 1
        return FakeAnnotatorResponse(sut_text=annotation_request.text)

    def translate_response(self, request, response: FakeAnnotatorResponse) -> SafetyAnnotation:
        return SafetyAnnotation(
            is_safe=True,
            is_valid=True,
        )


class BadAnnotator(FakeSafetyAnnotator):
    """Fake annotator that always fails to annotate."""

    def annotate(self, annotation_request: FakeAnnotatorRequest):
        """Always raises an error when trying to annotate."""
        raise RuntimeError("Failed to annotate")
