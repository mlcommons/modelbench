import string

import pytest

from modelgauge.annotators.prompt_engineered import PromptEngineeredAnnotator
from modelgauge.model_options import ModelOptions
from modelgauge.prompt import ChatPrompt, TextPrompt
from modelgauge.prompt_formatting import format_chat
from modelgauge.sut import PromptResponseSUT, SUTResponse


class ParityCountSUT(PromptResponseSUT):
    """SUT returns 'safe' if the input has an even number of words else 'unsafe'."""

    def translate_text_prompt(self, prompt: TextPrompt, options: ModelOptions) -> str:
        return prompt.text

    def translate_chat_prompt(self, prompt: ChatPrompt, options: ModelOptions) -> str:
        return format_chat(prompt)

    def evaluate(self, request: str) -> str:
        word_parity = len(request.split()) % 2
        return "safe" if word_parity == 0 else "unsafe"

    def translate_response(self, request: str, response: str) -> SUTResponse:
        return SUTResponse(text=response)


@pytest.fixture
def annotator(isolated_suts):
    isolated_suts.register(ParityCountSUT, "parity_count")
    return PromptEngineeredAnnotator(
        uid="test_annotator",
        prompt_template=string.Template("$prompt\n$response"),
        sut_id="parity_count",
    )


def test_prompt_engineered_annotator_safe(annotator):
    prompt = TextPrompt(text="even prompt plus response is safe")  # 6
    response = SUTResponse(text="be safe")  # 2
    annotation = annotator.process(prompt, response)
    assert annotation.is_safe


def test_prompt_engineered_annotator_unsafe(annotator):
    prompt = TextPrompt(text="odd prompt plus response not safe")  # 6
    response = SUTResponse(text="safe?")  # 1
    annotation = annotator.process(prompt, response)
    assert not annotation.is_safe


def test_prompt_engineered_bad_template():
    with pytest.raises(ValueError, match="Prompt template may only have 'prompt' and 'response' placeholders"):
        PromptEngineeredAnnotator(
            uid="bad_template",
            prompt_template=string.Template("This template is missing placeholders"),
            sut_id="parity_count",
        )


def test_prompt_engineered_response_only(isolated_suts):
    isolated_suts.register(ParityCountSUT, "parity_count")
    annotator = PromptEngineeredAnnotator(
        uid="response_only",
        prompt_template=string.Template("$response"),
        sut_id="parity_count",
    )
    prompt = TextPrompt(text="even prompt but ignored")  # 5
    response = SUTResponse(text="odd is unsafe")  # 3
    annotation = annotator.process(prompt, response)
    assert not annotation.is_safe


def test_prompt_engineered_prompt_only():
    with pytest.raises(
        ValueError,
        match="Prompt template may only have 'prompt' and 'response' placeholders, where 'response' is required",
    ):
        PromptEngineeredAnnotator(
            uid="prompt_only",
            prompt_template=string.Template("$prompt"),
            sut_id="parity_count",
        )
