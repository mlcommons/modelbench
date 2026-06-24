from typing import Optional

from modelgauge.annotation import SafetyAnnotation
from modelgauge.annotator import Annotator
from modelgauge.annotators.request import AnnotationRequest, AnnotatorSideInformation
from modelgauge.prompt import ChatPrompt, TextPrompt
from modelgauge.sut import SUTResponse


class SideInformationAwareAnnotator(Annotator):
    """Abstract Annotator that can accept side information."""

    def translate_prompt(
        self,
        prompt: TextPrompt | ChatPrompt,
        response: SUTResponse,
        side_information: Optional[AnnotatorSideInformation] = None,
    ) -> AnnotationRequest:
        if not isinstance(prompt, TextPrompt):
            raise ValueError(f"{self.__class__.__name__} only supports TextPrompt")
        return AnnotationRequest(
            annotator=self.uid,
            prompt=prompt.text,
            response=response.text,
            side_information=side_information.info if side_information else {},
        )

    def process(
        self,
        prompt: TextPrompt | ChatPrompt,
        response: SUTResponse,
        side_information: Optional[AnnotatorSideInformation] = None,
    ) -> SafetyAnnotation:
        # Proper fix in the future should port this signature to Annotator.process
        # `translate_prompt` should be updated to allow side information.
        annotator_request = self.translate_prompt(prompt, response, side_information)
        annotator_response = self.annotate(annotator_request)
        return self.translate_response(annotator_request, annotator_response)
