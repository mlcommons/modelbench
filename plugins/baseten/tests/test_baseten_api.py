import pytest

from modelgauge.sut import SUTOptions, SUTResponse
from modelgauge.suts.baseten_api import (
    BasetenPromptSUT,
    BasetenMessagesSUT,
    BasetenInferenceAPIKey,
    BasetenChatPromptRequest,
    BasetenChatMessagesRequest,
    BasetenChatMessage,
    BasetenResponse,
)
from modelgauge.prompt import TextPrompt
from modelgauge.typed_data import is_typeable


FAKE_MODEL_NAME = "xyzzy"


@pytest.fixture
def baseten_prompt_sut():
    return BasetenPromptSUT(
        "fake-sut",
        FAKE_MODEL_NAME,
        "https://model-FAKE_MODEL_NAME.api.baseten.co/production/predict",
        BasetenInferenceAPIKey("fake-api-key"),
    )


@pytest.fixture
def baseten_messages_sut():
    return BasetenMessagesSUT(
        "fake-sut",
        FAKE_MODEL_NAME,
        "https://model-FAKE_MODEL_NAME.api.baseten.co/production/predict",
        BasetenInferenceAPIKey("fake-api-key"),
    )


def _make_chat_request(model_id, prompt_text, **sut_options):
    return BasetenChatMessagesRequest(
        model=model_id,
        messages=[BasetenChatMessage(role="user", content=prompt_text)],
        **sut_options,
    )


def _make_response(response_text):
    return BasetenResponse(
        id="id",
        object="chat.completion",
        created="123456789",
        model=FAKE_MODEL_NAME,
        choices=[{"index": 0, "message": {"role": "assistant", "content": response_text}}],
        usage={},
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
    assert request.max_tokens == 200
    assert request.temperature == 0.5
    assert request.top_p == 0.5
    assert request.top_k == 10
    assert request.frequency_penalty == 2


def test_can_cache_request():
    request = _make_chat_request(FAKE_MODEL_NAME, "some-text", max_tokens=100)
    assert is_typeable(request)


def test_can_cache_response():
    response = _make_response("response")
    assert is_typeable(response)


def test_translate_response(baseten_messages_sut):
    request = _make_chat_request(FAKE_MODEL_NAME, "some-text")
    response = _make_response("response")

    translated_response = baseten_messages_sut.translate_response(request, response)

    assert translated_response == SUTResponse(text="response")
