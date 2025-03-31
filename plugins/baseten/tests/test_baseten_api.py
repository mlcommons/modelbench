import pytest
from modelgauge.sut import SUTOptions
from modelgauge.suts.baseten_api import (
    BasetenPromptSUT,
    BasetenMessagesSUT,
    BasetenInferenceAPIKey,
    BasetenChatPromptRequest,
    BasetenChatMessagesRequest,
    BasetenChatMessage,
)
from modelgauge.prompt import TextPrompt

# TODO: This could be configured but that would only be useful if this were to make an actual API request
_model = "xyzzy"


@pytest.fixture
def baseten_prompt_sut():
    return BasetenPromptSUT(
        "fake-sut",
        f"https://model-{_model.strip()}.api.baseten.co/production/predict",
        BasetenInferenceAPIKey("fake-api-key"),
    )


@pytest.fixture
def baseten_messages_sut():
    return BasetenMessagesSUT(
        "fake-sut",
        f"https://model-{_model.strip()}.api.baseten.co/production/predict",
        BasetenInferenceAPIKey("fake-api-key"),
    )


def test_baseten_api_translate_prompt_options(baseten_prompt_sut):
    options = SUTOptions(max_tokens=200)
    q = "What is xyzzy?"
    prompt = TextPrompt(text=q)

    request = baseten_prompt_sut.translate_text_prompt(prompt, options=options)

    assert request.prompt == q
    assert request.max_tokens == 200


def test_baseten_api_translate_messages_options(baseten_messages_sut):
    options = SUTOptions(max_tokens=200, temperature=0.5, top_p=0.5, top_k_per_token=10, frequency_penalty=2)
    q = "What is xyzzy?"
    prompt = TextPrompt(text=q)

    request = baseten_messages_sut.translate_text_prompt(prompt, options=options)

    assert request.messages[0].content == q
    assert request.messages[0].role == "user"
    assert request.max_new_tokens == 200
    assert request.temperature == 0.5
    assert request.top_p == 0.5
    assert request.top_k == 10
    assert request.frequency_penalty == 2
