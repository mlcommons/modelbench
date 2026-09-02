from typing import Any

from pydantic import BaseModel

from modelgauge.annotator import Annotator
from modelgauge.annotators.request import AnnotationRequest
from modelgauge.prompt import TextPrompt
from modelgauge.single_turn_prompt_response import TestItem
from modelgauge.sut import _READINESS_CHECK_TEXT_PROMPT, SUTResponse


class AnnotationRequestAnnotator(Annotator):
    """Annotator whose native request type is AnnotationRequest."""

    READINESS_CHECK_TEST_ITEM = TestItem(
        prompt=_READINESS_CHECK_TEXT_PROMPT,
        source_id="ignored",
        context={"hazard": "dfm"},
    )

    def _coerce_context_to_dict(self, context: Any) -> dict:
        if context is None:
            return {}
        if isinstance(context, dict):
            return context
        if isinstance(context, BaseModel):
            return context.model_dump()
        return {}

    def translate_prompt(self, test_item: TestItem, response: SUTResponse) -> AnnotationRequest:
        prompt = test_item.evaluated_prompt
        assert isinstance(prompt, TextPrompt), f"{self.__class__.__name__} only supports TextPrompt"
        return AnnotationRequest(
            annotator=self.uid,
            prompt=prompt.text,
            response=response.text,
            side_information=self._coerce_context_to_dict(test_item.context),
        )
