from pydantic import BaseModel

from modelgauge.annotation import SafetyAnnotation
from modelgauge.annotators.annotation_request_annotator import AnnotationRequestAnnotator
from modelgauge.annotators.request import AnnotationRequest
from modelgauge.prompt import TextPrompt
from modelgauge.single_turn_prompt_response import TestItem
from modelgauge.sut import SUTResponse


class _FakeContext(BaseModel):
    hazard: str
    persona_type: str


class _FakeAnnotator(AnnotationRequestAnnotator):
    def annotate(self, annotation_request: AnnotationRequest):
        return annotation_request

    def translate_response(self, request, response) -> SafetyAnnotation:
        return SafetyAnnotation(is_safe=True, is_valid=True)


def _make_test_item(context) -> TestItem:
    return TestItem(prompt=TextPrompt(text="hello, world"), source_id="id", context=context)


def test_translate_request_with_dict_context():
    annotator = _FakeAnnotator("fake-annotation-request")
    test_item = _make_test_item({"hazard": "dfm"})

    request = annotator.translate_request(test_item, SUTResponse(text="a response"))

    assert request.side_information == {"hazard": "dfm"}


def test_translate_request_with_base_model_context():
    annotator = _FakeAnnotator("fake-annotation-request")
    context = _FakeContext(hazard="dfm", persona_type="normal")
    test_item = _make_test_item(context)

    request = annotator.translate_request(test_item, SUTResponse(text="a response"))

    assert request.side_information == {"hazard": "dfm", "persona_type": "normal"}


def test_translate_request_with_no_context():
    annotator = _FakeAnnotator("fake-annotation-request")
    test_item = _make_test_item(None)

    request = annotator.translate_request(test_item, SUTResponse(text="a response"))

    assert request.side_information == {}
