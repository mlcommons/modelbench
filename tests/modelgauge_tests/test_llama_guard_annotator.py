from modelgauge.annotators.llama_guard_annotator import (
    LlamaGuardAnnotation,
    LlamaGuardAnnotator,
    LlamaGuardVersion,
)
from modelgauge.prompt import TextPrompt
from modelgauge.single_turn_prompt_response import PromptWithContext
from modelgauge.sut import SUTCompletion
from modelgauge.suts.together_client import (
    TogetherApiKey,
    TogetherChatRequest,
    TogetherChatResponse,
    TogetherCompletionsResponse,
    TogetherLogProbs,
)
import pytest
from typing import List


def _make_response(text: str) -> TogetherCompletionsResponse:
    return TogetherCompletionsResponse(
        id="some-id",
        choices=[
            TogetherCompletionsResponse.Choice(
                text=text,
                logprobs=TogetherLogProbs(
                    tokens=text.split(),
                    token_logprobs=[-0.01, -0.02, -0.03],
                ),
            )
        ],
        usage=TogetherCompletionsResponse.Usage(
            prompt_tokens=11, completion_tokens=12, total_tokens=13
        ),
        created=99,
        model="some-model",
        object="some-object",
    )


def _annotate_test_item(annotator, completion_text):
    prompt = PromptWithContext(prompt=TextPrompt(text="The prompt"), source_id=None)
    completion = SUTCompletion(text=completion_text)
    # TODO: Remove this and test on actual interface
    request = annotator.translate_request(prompt, completion)
    response = annotator.annotate(request)
    return annotator.translate_response(request, response)


class MockTogetherSUT:
    def __init__(self, responses: List):
        self.response_stack = list(reversed(responses))
        self.requests_received: List[TogetherChatRequest] = []

    def evaluate(self, request: TogetherChatRequest) -> TogetherChatResponse:
        self.requests_received.append(request)
        return self.response_stack.pop()


@pytest.mark.parametrize(
    "llama_guard_version", [LlamaGuardVersion.VERSION_1, LlamaGuardVersion.VERSION_2]
)
def test_safe(llama_guard_version):
    responses = [_make_response("safe")]
    annotator = LlamaGuardAnnotator(
        "lg", TogetherApiKey("some-value"), llama_guard_version=llama_guard_version
    )
    annotator.model = MockTogetherSUT(responses)

    result = _annotate_test_item(annotator, "first")

    assert result == LlamaGuardAnnotation(
        is_safe=True, is_safe_logprob=-0.01, violation_categories=[]
    )


@pytest.mark.parametrize(
    "llama_guard_version,sut_response_category,unsafe_category_name",
    [
        (LlamaGuardVersion.VERSION_1, "O1", "Violence and Hate."),
        (LlamaGuardVersion.VERSION_2, "S10", "Self-Harm. "),
    ],
)
def test_unsafe(llama_guard_version, sut_response_category, unsafe_category_name):
    responses = [_make_response("unsafe\n" + sut_response_category)]
    annotator = LlamaGuardAnnotator(
        "lg", TogetherApiKey("some-value"), llama_guard_version=llama_guard_version
    )
    annotator.model = MockTogetherSUT(responses)

    result = _annotate_test_item(annotator, "first")

    assert result == LlamaGuardAnnotation(
        is_safe=False,
        is_safe_logprob=-0.01,
        violation_categories=[unsafe_category_name],
    )


@pytest.mark.parametrize(
    "llama_guard_version,sut_response_categories,unsafe_category_names",
    [
        (
            LlamaGuardVersion.VERSION_1,
            "O1,O2",
            ["Violence and Hate.", "Sexual Content."],
        ),
        (LlamaGuardVersion.VERSION_2, "S10,S11", ["Self-Harm. ", "Sexual Content. "]),
    ],
)
def test_unsafe_multiple_categories(
    llama_guard_version, sut_response_categories, unsafe_category_names
):
    responses = [_make_response("unsafe\n" + sut_response_categories)]
    annotator = LlamaGuardAnnotator(
        "lg", TogetherApiKey("some-value"), llama_guard_version=llama_guard_version
    )
    annotator.model = MockTogetherSUT(responses)

    result = _annotate_test_item(annotator, "first")

    assert result == LlamaGuardAnnotation(
        is_safe=False,
        is_safe_logprob=-0.01,
        violation_categories=unsafe_category_names,
    )


def test_custom_prompt_decoder():
    def prompt_formatter(prompt, llama_guard_version):
        return f"Is agent saying bad things?\nagent: {prompt}"

    decoder = {"foo": "bar"}
    responses = [_make_response("unsafe\nfoo")]
    annotator = LlamaGuardAnnotator(
        "lg", TogetherApiKey("some-value"), prompt_formatter, decoder
    )
    annotator.model = MockTogetherSUT(responses)

    result = _annotate_test_item(annotator, "first")

    assert result == LlamaGuardAnnotation(
        is_safe=False,
        is_safe_logprob=-0.01,
        violation_categories=["bar"],
    )
    assert (
        annotator.model.requests_received[0].prompt
        == "Is agent saying bad things?\nagent: first"
    )
