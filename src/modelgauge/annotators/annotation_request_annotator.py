from modelgauge.annotator import Annotator
from modelgauge.annotators.request import AnnotationRequest
from modelgauge.prompt import TextPrompt
from modelgauge.single_turn_prompt_response import TestItem
from modelgauge.sut import SUTResponse


class AnnotationRequestAnnotator(Annotator):
    """Annotator whose native request type is AnnotationRequest."""

    def translate_prompt(self, test_item: TestItem, response: SUTResponse) -> AnnotationRequest:
        prompt = test_item.evaluated_prompt
        assert isinstance(prompt, TextPrompt), f"{self.__class__.__name__} only supports TextPrompt"
        side_information = test_item.context if isinstance(test_item.context, dict) else {}
        return AnnotationRequest(
            annotator=self.uid,
            prompt=prompt.text,
            response=response.text,
            side_information=side_information,
        )
