import pytest
from huggingface_hub import ChatCompletionOutputLogprob, ChatCompletionOutputLogprobs, ChatCompletionOutputTopLogprob, InferenceEndpointStatus  # type: ignore
from huggingface_hub.utils import HfHubHTTPError  # type: ignore
from pydantic import BaseModel
from typing import Optional
from unittest.mock import Mock, patch

from modelgauge.auth.huggingface_inference_token import HuggingFaceInferenceToken
from modelgauge.prompt import SUTOptions, TextPrompt
from modelgauge.sut import SUTCompletion, SUTResponse, TokenProbability, TopTokens
from modelgauge.suts.huggingface_chat_completion import (
    ChatMessage,
    HuggingFaceChatCompletionRequest,
    HuggingFaceChatCompletionSUT,
)


@pytest.fixture
def mock_endpoint():
    mock_endpoint = Mock()
    mock_endpoint.status = InferenceEndpointStatus.RUNNING
    mock_endpoint.url = "https://www.example.com"
    return mock_endpoint


@pytest.fixture
@patch("modelgauge.suts.huggingface_chat_completion.get_inference_endpoint")
def fake_sut(mock_get_inference_endpoint, mock_endpoint):
    mock_get_inference_endpoint.return_value = mock_endpoint

    sut = HuggingFaceChatCompletionSUT("fake_uid", "fake_endpoint", HuggingFaceInferenceToken("fake_token"))
    return sut


def _make_prompt(top_logprobs=None):
    extra_options = {}
    if top_logprobs is not None:
        extra_options["top_logprobs"] = top_logprobs
    return TextPrompt(
        text="some text prompt",
        options=SUTOptions(max_tokens=5, temperature=1.0, random="random", **extra_options),
    )


def _make_sut_request(top_logprobs: Optional[int] = None):
    extra_options = {}
    if top_logprobs is not None:
        extra_options["top_logprobs"] = top_logprobs
    return HuggingFaceChatCompletionRequest(
        messages=[ChatMessage(role="user", content="some text prompt")],
        logprobs=top_logprobs is not None,
        max_tokens=5,
        temperature=1.0,
        **extra_options,
    )


@pytest.mark.parametrize("top_logprobs", [None, 2])
def test_huggingface_chat_completion_translate_text_prompt_request(fake_sut, top_logprobs):
    prompt = _make_prompt(top_logprobs)

    request = fake_sut.translate_text_prompt(prompt)

    assert isinstance(request, HuggingFaceChatCompletionRequest)
    assert request == _make_sut_request(top_logprobs)


@pytest.mark.parametrize(
    "endpoint_status",
    [
        InferenceEndpointStatus.PENDING,
        InferenceEndpointStatus.INITIALIZING,
        InferenceEndpointStatus.UPDATING,
    ],
)
@patch("modelgauge.suts.huggingface_chat_completion.get_inference_endpoint")
def test_huggingface_chat_completion_connect_endpoint(
    mock_get_inference_endpoint, fake_sut, mock_endpoint, endpoint_status
):
    mock_get_inference_endpoint.return_value = mock_endpoint
    mock_endpoint.status = endpoint_status

    fake_sut._create_client()
    mock_endpoint.wait.assert_called_once()


@patch("modelgauge.suts.huggingface_chat_completion.get_inference_endpoint")
def test_huggingface_chat_completion_connect_endpoint_scaled_to_zero(
    mock_get_inference_endpoint, fake_sut, mock_endpoint
):
    mock_get_inference_endpoint.return_value = mock_endpoint
    mock_endpoint.status = InferenceEndpointStatus.SCALED_TO_ZERO

    fake_sut._create_client()

    mock_endpoint.resume.assert_called_once()
    mock_endpoint.wait.assert_called_once()


@patch("modelgauge.suts.huggingface_chat_completion.get_inference_endpoint")
def test_huggingface_chat_completion_connect_endpoint_fails_to_resume(
    mock_get_inference_endpoint, fake_sut, mock_endpoint
):
    mock_get_inference_endpoint.return_value = mock_endpoint
    mock_endpoint.status = InferenceEndpointStatus.SCALED_TO_ZERO
    mock_endpoint.resume.side_effect = HfHubHTTPError("Failure.")

    with pytest.raises(ConnectionError, match="Failed to resume endpoint. Please resume manually."):
        fake_sut._create_client()
        mock_endpoint.wait.assert_not_called()


@patch("modelgauge.suts.huggingface_chat_completion.get_inference_endpoint")
def test_huggingface_chat_completion_connect_failed_endpoint(mock_get_inference_endpoint, fake_sut, mock_endpoint):
    mock_get_inference_endpoint.return_value = mock_endpoint
    mock_endpoint.status = InferenceEndpointStatus.FAILED

    with pytest.raises(ConnectionError):
        fake_sut._create_client()


@patch("modelgauge.suts.huggingface_chat_completion.get_inference_endpoint")
@patch("modelgauge.suts.huggingface_chat_completion.InferenceClient")
def test_huggingface_chat_completion_lazy_load_client(
    mock_client, mock_get_inference_endpoint, fake_sut, mock_endpoint
):
    sut_request = _make_sut_request()
    mock_get_inference_endpoint.return_value = mock_endpoint
    assert fake_sut.client is None

    fake_sut.evaluate(sut_request)

    mock_client.assert_called_with(base_url="https://www.example.com", token="fake_token")
    assert fake_sut.client is not None


@patch("modelgauge.suts.huggingface_chat_completion.InferenceClient")
def test_huggingface_chat_completion_evaluate(mock_client, fake_sut):
    sut_request = _make_sut_request()
    fake_sut.client = mock_client

    fake_sut.evaluate(sut_request)

    mock_client.chat_completion.assert_called_with(
        **{
            "messages": [{"content": "some text prompt", "role": "user"}],
            "logprobs": False,
            "max_tokens": 5,
            "temperature": 1.0,
        }
    )


@patch("modelgauge.suts.huggingface_chat_completion.InferenceClient")
def test_huggingface_chat_completion_evaluate_with_logprobs(mock_client, fake_sut):
    sut_request = _make_sut_request(top_logprobs=2)
    fake_sut.client = mock_client

    fake_sut.evaluate(sut_request)

    mock_client.chat_completion.assert_called_with(
        **{
            "messages": [{"content": "some text prompt", "role": "user"}],
            "logprobs": True,
            "top_logprobs": 2,
            "max_tokens": 5,
            "temperature": 1.0,
        }
    )


class FakeChoice(BaseModel):
    message: ChatMessage
    logprobs: Optional[ChatCompletionOutputLogprobs] = None


class FakeResponse(BaseModel):
    choices: list[FakeChoice]


def test_huggingface_chat_completion_translate_response(fake_sut):
    sut_request = _make_sut_request()
    evaluate_output = FakeResponse(choices=[FakeChoice(message=ChatMessage(content="response", role="assistant"))])

    response = fake_sut.translate_response(sut_request, evaluate_output)

    assert response == SUTResponse(completions=[SUTCompletion(text="response")])


def test_huggingface_chat_completion_translate_response_with_logprobs(fake_sut):
    sut_request = _make_sut_request(top_logprobs=2)
    logprobs_output = ChatCompletionOutputLogprobs(
        content=[
            ChatCompletionOutputLogprob(
                token="hello",
                logprob=-0.1,
                top_logprobs=[
                    ChatCompletionOutputTopLogprob(token="hello", logprob=-0.1),
                    ChatCompletionOutputTopLogprob(token="hola", logprob=-0.2),
                ],
            ),
            ChatCompletionOutputLogprob(
                token="world",
                logprob=-0.3,
                top_logprobs=[
                    ChatCompletionOutputTopLogprob(token="world", logprob=-0.3),
                    ChatCompletionOutputTopLogprob(token="peeps", logprob=-0.4),
                ],
            ),
        ]
    )

    evaluate_output = FakeResponse(
        choices=[FakeChoice(message=ChatMessage(content="hello world", role="assistant"), logprobs=logprobs_output)]
    )

    response = fake_sut.translate_response(sut_request, evaluate_output)

    assert response == SUTResponse(
        completions=[
            SUTCompletion(
                text="hello world",
                top_logprobs=[
                    TopTokens(
                        top_tokens=[
                            TokenProbability(token="hello", logprob=-0.1),
                            TokenProbability(token="hola", logprob=-0.2),
                        ]
                    ),
                    TopTokens(
                        top_tokens=[
                            TokenProbability(token="world", logprob=-0.3),
                            TokenProbability(token="peeps", logprob=-0.4),
                        ]
                    ),
                ],
            )
        ]
    )
