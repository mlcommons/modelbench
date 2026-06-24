from pydantic import BaseModel

from modelgauge.annotators.composer.context import EvalContext
from modelgauge.prompt import TextPrompt
from modelgauge.sut import SUTResponse


class AnnotatorSideInformation(BaseModel, frozen=True):
    """Side information for the annotator."""

    info: dict


class AnnotationRequest(BaseModel):
    annotator: str
    prompt: str = ""
    response: str
    side_information: dict = {}  # optional side information to pass to evaluator

    def get_text_prompt(self) -> TextPrompt:
        return TextPrompt(text=self.prompt)

    def get_sut_response(self) -> SUTResponse:
        return SUTResponse(text=self.response)

    def get_annotator_side_information(self) -> AnnotatorSideInformation:
        return AnnotatorSideInformation(info=self.side_information)

    def get_eval_context(self) -> EvalContext:
        return EvalContext(
            prompt=self.prompt,
            response=self.response,
            metadata=self.side_information,
        )
