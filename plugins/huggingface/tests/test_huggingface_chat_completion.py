import pytest
from huggingface_hub import (
    ChatCompletionOutput,
    ChatCompletionOutputComplete,
    ChatCompletionOutputLogprob,
    ChatCompletionOutputLogprobs,
    ChatCompletionOutputMessage,
    ChatCompletionOutputTopLogprob,
    ChatCompletionOutputUsage,
    InferenceEndpointStatus,
)  # type: ignore
from huggingface_hub.utils import HfHubHTTPError  # type: ignore
from typing import Optional
from unittest.mock import Mock, patch

from modelgauge.auth.huggingface_inference_token import HuggingFaceInferenceToken
from modelgauge.prompt import TextPrompt
from modelgauge.sut import SUTOptions, SUTResponse, TokenProbability, TopTokens
from modelgauge.suts.huggingface_chat_completion import (
    ChatMessage,
    HuggingFaceChatCompletionOutput,
    HuggingFaceChatCompletionRequest,
    HuggingFaceChatCompletionDedicatedSUT,
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

    sut = HuggingFaceChatCompletionDedicatedSUT("fake_uid", "fake_endpoint", HuggingFaceInferenceToken("fake_token"))
    return sut


def _make_sut_options(top_logprobs=None):
    extra_options = {}
    if top_logprobs is not None:
        extra_options["top_logprobs"] = top_logprobs
    return SUTOptions(max_tokens=5, temperature=1.0, random="random", **extra_options)


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
    request = fake_sut.translate_text_prompt(TextPrompt(text="some text prompt"), _make_sut_options(top_logprobs))

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

    try:
        fake_sut.evaluate(sut_request)
    except:
        # This is expected to fail since the mock client doesn't return anything
        pass

    mock_client.assert_called_with(base_url="https://www.example.com", token="fake_token")
    assert fake_sut.client is not None


@patch("modelgauge.suts.huggingface_chat_completion.InferenceClient")
def test_huggingface_chat_completion_evaluate_sends_correct_request_params(mock_client, fake_sut):
    sut_request = _make_sut_request()
    fake_sut.client = mock_client

    try:
        fake_sut.evaluate(sut_request)
    except:
        # This is expected to fail since the mock client doesn't return anything
        pass

    mock_client.chat_completion.assert_called_with(
        **{
            "messages": [{"content": "some text prompt", "role": "user"}],
            "logprobs": False,
            "max_tokens": 5,
            "temperature": 1.0,
        }
    )


@patch("modelgauge.suts.huggingface_chat_completion.InferenceClient")
def test_huggingface_chat_completion_evaluate_with_logprobs_sends_correct_request_params(mock_client, fake_sut):
    sut_request = _make_sut_request(top_logprobs=2)
    fake_sut.client = mock_client

    try:
        fake_sut.evaluate(sut_request)
    except:
        # This is expected to fail since the mock client doesn't return anything
        pass

    mock_client.chat_completion.assert_called_with(
        **{
            "messages": [{"content": "some text prompt", "role": "user"}],
            "logprobs": True,
            "top_logprobs": 2,
            "max_tokens": 5,
            "temperature": 1.0,
        }
    )


@patch("modelgauge.suts.huggingface_chat_completion.InferenceClient")
def test_huggingface_chat_completion_evaluate_transforms_response_correctly(mock_client, fake_sut):
    """The SUT correctly translates the chat_completion's dataclass response object into a cacheable version."""
    sut_request = _make_sut_request()
    fake_choice = ChatCompletionOutputComplete(
        finish_reason="stop",
        index=0,
        message=ChatCompletionOutputMessage(role="assistant", content="response", tool_calls=None),
        logprobs=None,
    )
    client_response = ChatCompletionOutput(
        choices=[fake_choice],
        created=10,
        id="id",
        model="fake-model",
        system_fingerprint="fingerprint",
        usage=ChatCompletionOutputUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0),
    )
    mock_client.chat_completion.return_value = client_response
    fake_sut.client = mock_client

    response = fake_sut.evaluate(sut_request)

    assert response == HuggingFaceChatCompletionOutput(
        choices=[
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"role": "assistant", "content": "response", "tool_calls": None},
                "logprobs": None,
            }
        ],
        created=10,
        id="id",
        model="fake-model",
        system_fingerprint="fingerprint",
        usage={"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
    )


@patch("modelgauge.suts.huggingface_chat_completion.InferenceClient")
def test_huggingface_chat_completion_evaluate_transforms_response_logprobs_correctly(mock_client, fake_sut):
    """The SUT correctly translates the chat_completion's dataclass response object into a cacheable version."""
    sut_request = _make_sut_request()
    logprobs = ChatCompletionOutputLogprobs(
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
    fake_choice = ChatCompletionOutputComplete(
        finish_reason="stop",
        index=0,
        message=ChatCompletionOutputMessage(role="assistant", content="hello world", tool_calls=None),
        logprobs=logprobs,
    )
    client_response = ChatCompletionOutput(
        choices=[fake_choice],
        created=10,
        id="id",
        model="fake-model",
        system_fingerprint="fingerprint",
        usage=ChatCompletionOutputUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0),
    )
    mock_client.chat_completion.return_value = client_response
    fake_sut.client = mock_client

    response = fake_sut.evaluate(sut_request)

    assert response.choices[0]["logprobs"] == {
        "content": [
            {
                "token": "hello",
                "logprob": -0.1,
                "top_logprobs": [{"token": "hello", "logprob": -0.1}, {"token": "hola", "logprob": -0.2}],
            },
            {
                "token": "world",
                "logprob": -0.3,
                "top_logprobs": [{"token": "world", "logprob": -0.3}, {"token": "peeps", "logprob": -0.4}],
            },
        ]
    }


def _make_huggingface_chat_completion_output(text, logprobs=None):
    return HuggingFaceChatCompletionOutput(
        choices=[
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"role": "assistant", "content": text, "tool_calls": None},
                "logprobs": logprobs,
            }
        ],
    )


def test_huggingface_chat_completion_translate_response(fake_sut):
    sut_request = _make_sut_request()
    evaluate_output = _make_huggingface_chat_completion_output("response")

    response = fake_sut.translate_response(sut_request, evaluate_output)

    assert response == SUTResponse(text="response")


def test_huggingface_chat_completion_translate_response_with_logprobs(fake_sut):
    sut_request = _make_sut_request(top_logprobs=2)
    logprobs_output = {
        "content": [
            {
                "token": "hello",
                "logprob": -0.1,
                "top_logprobs": [{"token": "hello", "logprob": -0.1}, {"token": "hola", "logprob": -0.2}],
            },
            {
                "token": "world",
                "logprob": -0.3,
                "top_logprobs": [{"token": "world", "logprob": -0.3}, {"token": "peeps", "logprob": -0.4}],
            },
        ]
    }
    evaluate_output = _make_huggingface_chat_completion_output("hello world", logprobs_output)

    response = fake_sut.translate_response(sut_request, evaluate_output)

    assert response == SUTResponse(
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
