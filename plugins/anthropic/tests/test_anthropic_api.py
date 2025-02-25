import pytest
from anthropic.types.message import Message as AnthropicMessage
from unittest.mock import patch

from modelgauge.general import APIException
from modelgauge.prompt import SUTOptions, TextPrompt
from modelgauge.sut import SUTResponse

from modelgauge.suts.anthropic_api import AnthropicRequest, AnthropicApiKey, AnthropicSUT
from modelgauge.suts.openai_client import OpenAIChatMessage
from modelgauge.typed_data import is_typeable


@pytest.fixture
def fake_sut():
    return AnthropicSUT("fake-sut", "fake-model", AnthropicApiKey("fake-api-key"))


@pytest.fixture
def simple_anthropic_request():
    return AnthropicRequest(model="fake-model", messages=[OpenAIChatMessage(content="some-text", role="user")])


def test_anthropic_api_translate_request_default_sut_options(fake_sut):
    prompt = TextPrompt(text="some-text")

    request = fake_sut.translate_text_prompt(prompt)

    assert isinstance(request, AnthropicRequest)
    assert request.model == "fake-model"
    assert request.messages == [OpenAIChatMessage(content="some-text", role="user")]
    assert request.max_tokens == 100  # Default SUTOptions value

    # Make sure all other attributes are not set
    request_dict = request.model_dump(exclude_none=False)
    for key in request_dict:
        if key not in ["model", "messages", "max_tokens"]:
            assert request_dict[key] is None


def test_anthropic_api_translate_request_non_default_sut_options(fake_sut):
    """Test that all possible generation parameters are set correctly."""
    options = SUTOptions(
        max_tokens=200,  # Overwrite default value
        temperature=0.5,
        top_k_per_token=10,
        stop_sequences=["stop"],
        top_p=0.5,
    )
    prompt = TextPrompt(text="some-text", options=options)

    request = fake_sut.translate_text_prompt(prompt)

    assert request.max_tokens == 200
    assert request.temperature == 0.5
    assert request.top_k == 10
    assert request.stop_sequences == ["stop"]
    assert request.top_p == 0.5


def test_can_cache_anthropic_api_request(simple_anthropic_request):
    assert is_typeable(simple_anthropic_request)


def test_can_cache_anthropic_api_response():
    response = AnthropicMessage(
        id="fake-id",
        content=[{"text": "response", "type": "text"}],
        model="fake-model",
        role="assistant",
        type="message",
        usage={"input_tokens": 0, "output_tokens": 0},
    )

    assert is_typeable(response)


@patch("modelgauge.suts.anthropic_api.Anthropic")
def test_anthropic_api_lazy_load_client(mock_client, fake_sut, simple_anthropic_request):
    assert fake_sut.client is None

    fake_sut.evaluate(simple_anthropic_request)

    assert fake_sut.client is not None


@patch("modelgauge.suts.anthropic_api.Anthropic")
def test_anthropic_api_evaluate_sends_correct_params(mock_client, fake_sut):
    fake_sut.client = mock_client
    request = AnthropicRequest(
        model="fake-model", messages=[OpenAIChatMessage(content="some-text", role="user")], top_k=10
    )

    fake_sut.evaluate(request)

    mock_client.messages.create.assert_called_with(
        model="fake-model", messages=[{"content": "some-text", "role": "user"}], top_k=10
    )


@patch("modelgauge.suts.anthropic_api.Anthropic")
def test_anthropic_api_evaluate_raises_api_exceptions(mock_client, fake_sut, simple_anthropic_request):
    mock_client.messages.create.side_effect = APIException("mocked error")
    fake_sut.client = mock_client

    with pytest.raises(APIException, match="mocked error"):
        fake_sut.evaluate(simple_anthropic_request)


def test_anthropic_api_translate_response(fake_sut, simple_anthropic_request):
    fake_response = AnthropicMessage(
        id="fake-id",
        content=[{"text": "response", "type": "text"}],
        model="fake-model",
        role="assistant",
        type="message",
        usage={"input_tokens": 1, "output_tokens": 1},
    )

    translated_response = fake_sut.translate_response(simple_anthropic_request, fake_response)

    assert translated_response == SUTResponse(text="response")
