from typing import List
from newhelm.annotators.llama_guard_annotator import (
    LlamaGuardAnnotation,
    LlamaGuardAnnotator,
)
from newhelm.suts.together_client import (
    TogetherApiKey,
    TogetherChatRequest,
    TogetherChatResponse,
    TogetherCompletionsResponse,
)
from newhelm.prompt import TextPrompt
from newhelm.single_turn_prompt_response import PromptInteraction, PromptWithContext

from newhelm.sut import SUTCompletion, SUTResponse

Interaction = LlamaGuardAnnotation.Interaction
Completion = LlamaGuardAnnotation.Completion


def _make_interaction(completions: List[str]) -> PromptInteraction:
    sut_completions = [SUTCompletion(text=text) for text in completions]
    return PromptInteraction(
        prompt=PromptWithContext(prompt=TextPrompt(text="The prompt"), source_id=None),
        response=SUTResponse(completions=sut_completions),
    )


def _make_response(text: str) -> TogetherCompletionsResponse:
    return TogetherCompletionsResponse(
        id="some-id",
        choices=[TogetherCompletionsResponse.Choice(text=text)],
        usage=TogetherCompletionsResponse.Usage(
            prompt_tokens=11, completion_tokens=12, total_tokens=13
        ),
        created=99,
        model="some-model",
        object="some-object",
    )


class MockTogetherSUT:
    def __init__(self, responses: List):
        self.response_stack = list(reversed(responses))
        self.requests_received: List[TogetherChatRequest] = []

    def evaluate(self, request: TogetherChatRequest) -> TogetherChatResponse:
        self.requests_received.append(request)
        return self.response_stack.pop()


def test_safe():
    interactions = [_make_interaction(["first"])]
    responses = [_make_response("safe")]
    annotator = LlamaGuardAnnotator(TogetherApiKey("some-value"))
    annotator.model = MockTogetherSUT(responses)

    result = annotator.annotate_test_item(interactions)

    assert result == LlamaGuardAnnotation(
        interactions=[
            Interaction(completions=[Completion(is_safe=True, violation_categories=[])])
        ]
    )


def test_unsafe():
    interactions = [_make_interaction(["first"])]
    responses = [_make_response("unsafe\nO1")]
    annotator = LlamaGuardAnnotator(TogetherApiKey("some-value"))
    annotator.model = MockTogetherSUT(responses)

    result = annotator.annotate_test_item(interactions)

    assert result == LlamaGuardAnnotation(
        interactions=[
            Interaction(
                completions=[
                    Completion(
                        is_safe=False, violation_categories=["Violence and Hate"]
                    )
                ]
            )
        ]
    )


def test_unsafe_multiple_categories():
    interactions = [_make_interaction(["first"])]
    responses = [_make_response("unsafe\nO1,O2")]
    annotator = LlamaGuardAnnotator(TogetherApiKey("some-value"))
    annotator.model = MockTogetherSUT(responses)

    result = annotator.annotate_test_item(interactions)

    assert result == LlamaGuardAnnotation(
        interactions=[
            Interaction(
                completions=[
                    Completion(
                        is_safe=False,
                        violation_categories=["Violence and Hate", "Sexual Content"],
                    )
                ]
            )
        ]
    )


def test_custom_prompt_decoder():
    def prompt_formatter(user, prompt):
        return f"Is {user} saying bad things? {prompt}"

    decoder = {"foo": "bar"}
    interactions = [_make_interaction(["first"])]
    responses = [_make_response("unsafe\nfoo")]
    annotator = LlamaGuardAnnotator(
        TogetherApiKey("some-value"), prompt_formatter, decoder
    )
    annotator.model = MockTogetherSUT(responses)

    result = annotator.annotate_test_item(interactions)

    assert result == LlamaGuardAnnotation(
        interactions=[
            Interaction(
                completions=[
                    Completion(
                        is_safe=False,
                        violation_categories=["bar"],
                    )
                ]
            )
        ],
    )
    assert (
        annotator.model.requests_received[0].prompt
        == "Is assistant saying bad things? assistant: first"
    )
