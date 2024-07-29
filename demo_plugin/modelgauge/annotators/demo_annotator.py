from modelgauge.annotator import CompletionAnnotator
from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.single_turn_prompt_response import PromptWithContext
from modelgauge.sut import SUTCompletion
from pydantic import BaseModel


class DemoYBadAnnotation(BaseModel):
    """How bad each SUTResponse in the TestItem is."""

    badness: float


class DemoYBadRequest(BaseModel):
    text: str


class DemoYBadResponse(BaseModel):
    score: float


class DemoYBadAnnotator(CompletionAnnotator[DemoYBadAnnotation]):
    """A demonstration annotator that dislikes the letter Y.

    Real Annotators are intended to do expensive processing on the string,
    such as calling another model or collecting data from human raters. For
    the demo though, we want something cheap and deterministic.
    """

    def translate_request(self, prompt: PromptWithContext, completion: SUTCompletion):
        return DemoYBadRequest(text=completion.text)

    def annotate(self, annotation_request: DemoYBadRequest) -> DemoYBadResponse:
        score = 0
        for character in annotation_request.text:
            if character in {"Y", "y"}:
                score += 1
        return DemoYBadResponse(score=score)

    def translate_response(
        self, request, response: DemoYBadResponse
    ) -> DemoYBadAnnotation:
        return DemoYBadAnnotation(badness=response.score)


ANNOTATORS.register(DemoYBadAnnotator, "demo_annotator")
