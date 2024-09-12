import pytest
from modelgauge.annotators.openai_compliance_annotator import (
    ComplianceAnnotation,
    OpenAIComplianceAnnotator,
)
from modelgauge.prompt import ChatMessage, ChatPrompt, TextPrompt
from modelgauge.single_turn_prompt_response import PromptWithContext
from modelgauge.sut import SUTCompletion, SUTResponse
from modelgauge.suts.openai_client import OpenAIApiKey, OpenAIOrgId
from typing import List


def _make_annotation(compliance: str) -> ComplianceAnnotation:
    return ComplianceAnnotation(compliance=compliance)


def _annotate_test_item(
    annotator, prompt=TextPrompt(text="The prompt"), completion_text=""
):
    # TODO: Remove this and test on actual interface
    request = annotator.translate_request(
        PromptWithContext(prompt=prompt, source_id=None),
        SUTCompletion(text=completion_text),
    )
    response = annotator.annotate(request)
    return annotator.translate_response(request, response)


class MockSUT:
    def __init__(self, responses: List):
        self.response_stack = list(reversed(responses))
        self.requests_received: List[str] = []

    def evaluate(self, request: str) -> str:
        self.requests_received.append(request)
        return self.response_stack.pop()

    def translate_text_prompt(self, prompt: TextPrompt) -> str:
        return prompt.text

    def translate_response(self, request: str, response: str) -> SUTResponse:
        return SUTResponse(completions=[SUTCompletion(text=response)])


def test_default_decoder():
    responses = ["1_full_compliance", "2_full_refusal", "3_partial_refusal"]
    annotator = OpenAIComplianceAnnotator(
        openai_api_key=OpenAIApiKey("fake-key"), openai_api_org_id=OpenAIOrgId(None)
    )
    annotator.model = MockSUT(responses)

    assert _annotate_test_item(annotator) == _make_annotation("Full Compliance")
    assert _annotate_test_item(annotator) == _make_annotation("Full Refusal")
    assert _annotate_test_item(annotator) == _make_annotation("Partial Refusal")


def test_overridden_decoder_keys():
    LABEL_MAPPING = {
        "FC": "Full Compliance",
        "FR": "Full Refusal",
        "PR": "Partial Refusal",
    }
    responses = ["FC", "1_full_compliance"]
    annotator = OpenAIComplianceAnnotator(sut=MockSUT(responses), decoder=LABEL_MAPPING)

    assert _annotate_test_item(annotator, completion_text="first") == _make_annotation(
        "Full Compliance"
    )

    # assert default decoder key isn't used
    with pytest.raises(Exception, match="Unexpected response"):
        _annotate_test_item(annotator, completion_text="second")


def test_chat_prompt_assertion():
    """Checks that annotating and interaction with a ChatPrompt and the default formatter results in an assertion error"""
    responses = ["1_full_compliance"]
    annotator = OpenAIComplianceAnnotator(sut=MockSUT(responses))
    prompt = ChatPrompt(messages=[ChatMessage(text="chat", role="USER")])
    with pytest.raises(Exception, match="Please provide custom formatter"):
        _annotate_test_item(annotator, prompt=prompt)


def test_chat_prompt_overridden_formatter():
    """Checks that overriding the formatter allows you to annotate ChatPrompt interactions."""
    custom_formatter = lambda sut_prompt, sut_completion: "formatted prompt"
    responses = ["1_full_compliance"]
    annotator = OpenAIComplianceAnnotator(
        sut=MockSUT(responses), formatter=custom_formatter
    )
    prompt = ChatPrompt(messages=[ChatMessage(text="chat", role="USER")])
    assert _annotate_test_item(annotator, prompt=prompt) == _make_annotation(
        "Full Compliance"
    )
