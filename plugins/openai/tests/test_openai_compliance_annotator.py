from newhelm.suts.openai_client import OpenAIApiKey, OpenAIOrgId
import pytest
from typing import List

from newhelm.annotators.openai_compliance_annotator import (
    ComplianceAnnotation,
    OpenAIComplianceAnnotator,
)
from newhelm.prompt import ChatMessage, ChatPrompt, TextPrompt
from newhelm.single_turn_prompt_response import PromptInteraction, PromptWithContext
from newhelm.sut import SUTCompletion, SUTResponse

Interaction = ComplianceAnnotation.Interaction
Completion = ComplianceAnnotation.Completion


def _make_interaction(completions: List[str] = [""]) -> PromptInteraction:
    sut_completions = [SUTCompletion(text=text) for text in completions]
    return PromptInteraction(
        prompt=PromptWithContext(prompt=TextPrompt(text="The prompt"), source_id=None),
        response=SUTResponse(completions=sut_completions),
    )


def _make_annotation(compliance: str) -> ComplianceAnnotation:
    return ComplianceAnnotation(
        interactions=[Interaction(completions=[Completion(compliance=compliance)])]
    )


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

    assert annotator.annotate_test_item([_make_interaction()]) == _make_annotation(
        "Full Compliance"
    )
    assert annotator.annotate_test_item([_make_interaction()]) == _make_annotation(
        "Full Refusal"
    )
    assert annotator.annotate_test_item([_make_interaction()]) == _make_annotation(
        "Partial Refusal"
    )


def test_overridden_decoder_keys():
    LABEL_MAPPING = {
        "FC": "Full Compliance",
        "FR": "Full Refusal",
        "PR": "Partial Refusal",
    }
    responses = ["FC", "1_full_compliance"]
    annotator = OpenAIComplianceAnnotator(sut=MockSUT(responses), decoder=LABEL_MAPPING)

    assert annotator.annotate_test_item(
        [_make_interaction(["first"])]
    ) == _make_annotation("Full Compliance")

    # assert default decoder key isn't used
    with pytest.raises(Exception, match="Unexpected response"):
        annotator.annotate_test_item([_make_interaction(["second"])])


def test_chat_prompt_assertion():
    """Checks that annotating and interaction with a ChatPrompt and the default formatter results in an assertion error"""
    responses = ["1_full_compliance"]
    annotator = OpenAIComplianceAnnotator(sut=MockSUT(responses))
    prompt = ChatPrompt(messages=[ChatMessage(text="chat", role="USER")])
    interaction = PromptInteraction(
        prompt=PromptWithContext(prompt=prompt, source_id=None),
        response=SUTResponse(completions=[SUTCompletion(text="completion")]),
    )
    with pytest.raises(Exception, match="Please provide custom formatter"):
        annotator.annotate_test_item([interaction])


def test_chat_prompt_overridden_formatter():
    """Checks that overriding the formatter allows you to annotate ChatPrompt interactions."""
    custom_formatter = lambda sut_prompt, sut_completion: "formatted prompt"
    responses = ["1_full_compliance"]
    annotator = OpenAIComplianceAnnotator(
        sut=MockSUT(responses), formatter=custom_formatter
    )
    # annotator.model = MockSUT(responses)
    prompt = ChatPrompt(messages=[ChatMessage(text="chat", role="USER")])
    interaction = PromptInteraction(
        prompt=PromptWithContext(prompt=prompt, source_id=None),
        response=SUTResponse(completions=[SUTCompletion(text="completion")]),
    )
    assert annotator.annotate_test_item([interaction]) == _make_annotation(
        "Full Compliance"
    )
