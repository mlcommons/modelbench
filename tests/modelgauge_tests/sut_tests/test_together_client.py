from unittest.mock import patch, MagicMock

import pytest
from requests import HTTPError  # type:ignore
import json

from modelgauge.general import APIException
from modelgauge.prompt import ChatMessage, ChatPrompt, ChatRole, TextPrompt
from modelgauge.prompt_formatting import format_chat
from modelgauge.sut import SUTOptions, SUTResponse, TokenProbability, TopTokens
from modelgauge.suts.together_client import (
    TogetherApiKey,
    TogetherChatResponse,
    TogetherChatRequest,
    TogetherChatSUT,
    TogetherCompletionsResponse,
    TogetherCompletionsRequest,
    TogetherCompletionsSUT,
    TogetherDedicatedChatSUT,
    TogetherInferenceResponse,
    TogetherInferenceRequest,
    TogetherInferenceSUT,
)

TOGETHER_CHAT_RESPONSE_JSON = """\
{
    "id": "87ca703b9c6710af-ORD",
    "object": "chat.completion",
    "created": 1714510586,
    "model": "mistralai/Mixtral-8x7B-v0.1",
    "prompt": [],
    "choices": [
        {
            "finish_reason": "length",
            "logprobs": null,
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Some response"
            }
        }
    ],
    "usage": {
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "total_tokens": 7
    }
} 
"""


class MockResponse:
    """Bare bones mock of requests.Response"""

    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if 400 <= self.status_code < 500:
            raise HTTPError(f"Status {self.status_code}")


def _make_client(sut_class):
    return sut_class(
        uid="test-model",
        model="some-model",
        api_key=TogetherApiKey("some-value"),
    )


@pytest.mark.parametrize(
    "sut_class,request_class",
    [
        (TogetherCompletionsSUT, TogetherCompletionsRequest),
        (TogetherInferenceSUT, TogetherInferenceRequest),
    ],
)
def test_together_translate_text_prompt_request(sut_class, request_class):
    client = _make_client(sut_class)
    prompt = TextPrompt(text="some-text")
    request = client.translate_text_prompt(prompt, SUTOptions())
    assert request == request_class(
        model="some-model",
        prompt="some-text",
        max_tokens=100,
        n=1,
    )


@pytest.mark.parametrize(
    "sut_class,request_class",
    [
        (TogetherCompletionsSUT, TogetherCompletionsRequest),
        (TogetherInferenceSUT, TogetherInferenceRequest),
    ],
)
def test_together_translate_chat_prompt_request(sut_class, request_class):
    client = _make_client(sut_class)
    prompt = ChatPrompt(
        messages=[
            ChatMessage(text="some-text", role=ChatRole.user),
            ChatMessage(text="more-text", role=ChatRole.sut),
        ]
    )
    request = client.translate_chat_prompt(prompt, SUTOptions())
    assert request == request_class(
        model="some-model",
        prompt=format_chat(prompt, user_role="user", sut_role="assistant"),
        max_tokens=100,
        n=1,
    )


def test_together_chat_translate_text_prompt_request():
    client = _make_client(TogetherChatSUT)
    prompt = TextPrompt(text="some-text")
    request = client.translate_text_prompt(prompt, SUTOptions())
    assert request == TogetherChatRequest(
        model="some-model",
        messages=[TogetherChatRequest.Message(content="some-text", role="user")],
        max_tokens=100,
        n=1,
    )


def test_together_chat_translate_chat_prompt_request():
    client = _make_client(TogetherChatSUT)
    prompt = ChatPrompt(
        messages=[
            ChatMessage(text="some-text", role=ChatRole.user),
            ChatMessage(text="more-text", role=ChatRole.sut),
        ]
    )
    request = client.translate_chat_prompt(prompt, SUTOptions())
    assert request == TogetherChatRequest(
        model="some-model",
        messages=[
            TogetherChatRequest.Message(content="some-text", role="user"),
            TogetherChatRequest.Message(content="more-text", role="assistant"),
        ],
        max_tokens=100,
        n=1,
    )


@pytest.mark.parametrize(
    "sut_class,request_class",
    [
        (TogetherCompletionsSUT, TogetherCompletionsRequest),
        (TogetherInferenceSUT, TogetherInferenceRequest),
    ],
)
def test_together_translate_request_logprobs(sut_class, request_class):
    client = _make_client(sut_class)
    prompt = TextPrompt(text="some-text")
    request = client.translate_text_prompt(prompt, SUTOptions(top_logprobs=1))
    assert request == request_class(
        model="some-model",
        prompt="some-text",
        max_tokens=100,
        n=1,
        logprobs=1,
    )


def test_together_chat_translate_request_logprobs():
    client = _make_client(TogetherChatSUT)
    prompt = TextPrompt(text="some-text")
    request = client.translate_text_prompt(prompt, SUTOptions(top_logprobs=1))
    assert request == TogetherChatRequest(
        model="some-model",
        messages=[TogetherChatRequest.Message(content="some-text", role="user")],
        max_tokens=100,
        n=1,
        logprobs=1,
    )


def test_together_completions_translate_response():
    client = _make_client(TogetherCompletionsSUT)
    request = TogetherCompletionsRequest(
        model="some-model",
        prompt="My favorite colors are red and ",
        max_tokens=2,
    )
    response = TogetherCompletionsResponse.model_validate_json(
        """\
{
    "id": "87cc221c3b411064-ORD",
    "object": "text.completion",
    "created": 1714528358,
    "model": "mistralai/Mixtral-8x7B-v0.1",
    "prompt": [],
    "choices": [
        {
            "text": " blue.",
            "finish_reason": "length",
            "logprobs": null,
            "index": 0
        }
    ],
    "usage": {
        "prompt_tokens": 8,
        "completion_tokens": 2,
        "total_tokens": 10
    }
} 

"""
    )
    result = client.translate_response(request, response)
    assert result == SUTResponse(text=" blue.", top_logprobs=None)


def test_together_completions_translate_response_logprobs():
    client = _make_client(TogetherCompletionsSUT)
    request = TogetherCompletionsRequest(
        model="some-model",
        prompt="My favorite colors are red and ",
        max_tokens=2,
        logprobs=1,
    )
    response = TogetherCompletionsResponse.model_validate_json(
        """\
{
    "id": "87cc221c3b411064-ORD",
    "object": "text.completion",
    "created": 1714528358,
    "model": "mistralai/Mixtral-8x7B-v0.1",
    "prompt": [],
    "choices": [
        {
            "text": " blue.",
            "finish_reason": "length",
            "logprobs": {
                "token_ids": [
                    5045,
                    28723
                ],
                "tokens": [
                    " blue",
                    "."
                ],
                "token_logprobs": [
                    -1.9072266,
                    -0.703125

                ]
            },
            "index": 0
        }
    ],
    "usage": {
        "prompt_tokens": 8,
        "completion_tokens": 2,
        "total_tokens": 10
    }
} 
"""
    )
    result = client.translate_response(request, response)
    assert result == SUTResponse(
        text=" blue.",
        top_logprobs=[
            TopTokens(top_tokens=[TokenProbability(token=" blue", logprob=-1.9072266)]),
            TopTokens(top_tokens=[TokenProbability(token=".", logprob=-0.703125)]),
        ],
    )


def test_together_inference_translate_response():
    client = _make_client(TogetherInferenceSUT)
    request = TogetherInferenceRequest(
        model="some-model",
        prompt="My favorite colors are red and ",
        max_tokens=2,
    )
    response = TogetherInferenceResponse.model_validate_json(
        """\
{
    "id": "87cdcf226b121417-ORD",
    "status": "finished",
    "prompt": [
        "My favorite colors are red and "
    ],
    "model": "mistralai/Mixtral-8x7B-v0.1",
    "model_owner": "",
    "num_returns": 1,
    "args": {
        "model": "mistralai/Mixtral-8x7B-v0.1",
        "prompt": "My favorite colors are red and ",
        "max_tokens": 2
    },
    "subjobs": [],
    "usage": {
        "prompt_tokens": 8,
        "completion_tokens": 2,
        "total_tokens": 10,
        "duration": 197
    },
    "output": {
        "finish_reason": "length",
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": 2,
            "total_tokens": 10
        },
        "result_type": "language-model-inference",
        "choices": [
            {
                "text": " blue.",
                "finish_reason": "length",
                "token_ids": [
                    5045,
                    28723
                ]
            }
        ],
        "prompt": [
            {
                "text": "My favorite colors are red and ",
                "token_ids": [],
                "tokens": [],
                "token_logprobs": []
            }
        ]
    }
} 
"""
    )
    result = client.translate_response(request, response)
    assert result == SUTResponse(text=" blue.", top_logprobs=None)


def test_together_inference_translate_response_logprobs():
    client = _make_client(TogetherInferenceSUT)
    request = TogetherInferenceRequest(
        model="some-model",
        prompt="My favorite colors are red and ",
        max_tokens=2,
        logprobs=1,
    )
    response = TogetherInferenceResponse.model_validate_json(
        """\
{
    "id": "87cdcf226b121417-ORD",
    "status": "finished",
    "prompt": [
        "My favorite colors are red and "
    ],
    "model": "mistralai/Mixtral-8x7B-v0.1",
    "model_owner": "",
    "num_returns": 1,
    "args": {
        "model": "mistralai/Mixtral-8x7B-v0.1",
        "prompt": "My favorite colors are red and ",
        "max_tokens": 2,
        "logprobs": 1
    },
    "subjobs": [],
    "usage": {
        "prompt_tokens": 8,
        "completion_tokens": 2,
        "total_tokens": 10,
        "duration": 293
    },
    "output": {
        "finish_reason": "length",
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": 2,
            "total_tokens": 10
        },
        "result_type": "language-model-inference",
        "choices": [
            {
                "text": " blue.",
                "finish_reason": "length",
                "token_ids": [
                    5045,
                    28723
                ],
                "tokens": [
                    " blue",
                    "."
                ],
                "token_logprobs": [
                    -1.9072266,
                    -0.703125
                ]
            }
        ],
        "prompt": [
            {
                "text": "My favorite colors are red and ",
                "token_ids": [],
                "tokens": [],
                "token_logprobs": []
            }
        ]
    }
} 
"""
    )
    result = client.translate_response(request, response)
    assert result == SUTResponse(
        text=" blue.",
        top_logprobs=[
            TopTokens(top_tokens=[TokenProbability(token=" blue", logprob=-1.9072266)]),
            TopTokens(top_tokens=[TokenProbability(token=".", logprob=-0.703125)]),
        ],
    )


def test_together_chat_evaluate_normal():
    client = _make_client(TogetherChatSUT)
    with patch("modelgauge.suts.together_client._retrying_request") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_post.return_value = mock_response
        with pytest.raises(APIException) as exc_info:
            client.evaluate(MagicMock())
            assert "401" in exc_info.value


def test_together_chat_translate_response():
    client = _make_client(TogetherChatSUT)
    request = TogetherChatRequest(
        model="some-model",
        messages=[],
        max_tokens=2,
    )
    response = TogetherChatResponse.model_validate_json(TOGETHER_CHAT_RESPONSE_JSON)
    result = client.translate_response(request, response)
    assert result == SUTResponse(text="Some response", top_logprobs=None)


def test_together_chat_translate_response_logprobs():
    client = _make_client(TogetherChatSUT)
    request = TogetherChatRequest(
        model="some-model",
        messages=[],
        max_tokens=2,
        logprobs=1,
    )
    response = TogetherChatResponse.model_validate_json(
        """\
{
    "id": "87ca703b9c6710af-ORD",
    "object": "chat.completion",
    "created": 1714510586,
    "model": "mistralai/Mixtral-8x7B-v0.1",
    "prompt": [],
    "choices": [
        {
            "finish_reason": "length",
            "logprobs":  {"token_ids": [13,
      1976],
     "tokens": ["Some",
     "response"],
     "token_logprobs": [-0.55810547,
      -0.9326172]},
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Some response"
            }
        }
    ],
    "usage": {
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "total_tokens": 7
    }
} 
"""
    )
    result = client.translate_response(request, response)
    assert result == SUTResponse(
        text="Some response",
        top_logprobs=[
            TopTokens(top_tokens=[TokenProbability(token="Some", logprob=-0.55810547)]),
            TopTokens(top_tokens=[TokenProbability(token="response", logprob=-0.9326172)]),
        ],
    )


class TestTogetherDedicatedChatSUT:
    @pytest.fixture
    def sut(self):
        sut = TogetherDedicatedChatSUT(
            uid="test-model",
            model="some-model",
            api_key=TogetherApiKey("some-value"),
        )
        sut.endpoint_id = "test-endpoint-id"
        sut.endpoint_status = "STARTED"
        return sut

    @pytest.fixture
    def mock_endpoints_response(self):
        # Mock the endpoints list response
        mock_endpoints_response = MagicMock()
        mock_endpoints_response.json.return_value = {"data": [{"name": "some-model", "id": "test-endpoint-id"}]}
        return mock_endpoints_response

    @pytest.fixture
    def mock_chat_response(self):
        mock_chat_response = MagicMock()
        mock_chat_response.status_code = 200
        mock_chat_response.json.return_value = json.loads(TOGETHER_CHAT_RESPONSE_JSON)
        return mock_chat_response

    def test_together_dedicated_chat_sut_endpoint_id_and_status(self, mock_chat_response, mock_endpoints_response):
        with patch("modelgauge.suts.together_client._retrying_request") as mock_request:
            # Mock the endpoint status response
            mock_status_response = MagicMock()
            mock_status_response.json.return_value = {"state": "STARTED"}

            # Configure the mock to return different responses based on the URL
            def side_effect(url, headers, json_payload, method):
                if url.endswith("/endpoints"):
                    return mock_endpoints_response
                elif url.endswith("/test-endpoint-id"):
                    return mock_status_response
                elif url == TogetherChatSUT._CHAT_COMPLETIONS_URL:
                    return mock_chat_response
                return MagicMock()

            mock_request.side_effect = side_effect

            sut = TogetherDedicatedChatSUT(
                uid="test-model",
                model="some-model",
                api_key=TogetherApiKey("some-value"),
            )
            assert sut.endpoint_id is None
            assert sut.endpoint_status is None

            request = TogetherChatRequest(model="some-model", messages=[])
            sut.evaluate(request)

            assert sut.endpoint_id == "test-endpoint-id"
            assert sut.endpoint_status == "STARTED"

    def test_together_dedicated_chat_sut_no_endpoint_found(self, mock_endpoints_response):
        with patch("modelgauge.suts.together_client._retrying_request") as mock_request:
            # Mock the endpoints list response with no matching endpoint
            mock_request.return_value = mock_endpoints_response
            sut = TogetherDedicatedChatSUT(
                uid="test-model",
                model="non-existent-model",
                api_key=TogetherApiKey("some-value"),
            )
            request = TogetherChatRequest(model="non-existent-model", messages=[])
            with pytest.raises(APIException, match="No endpoint found for model non-existent-model"):
                sut.evaluate(request)

    def test_evaluate_endpoint_already_started(self, sut, mock_chat_response):
        with patch("modelgauge.suts.together_client._retrying_request") as mock_request:
            # Mock successful chat response
            mock_request.return_value = mock_chat_response

            request = TogetherChatRequest(model="some-model", messages=[])
            response = sut.evaluate(request)

            assert isinstance(response, TogetherChatResponse)
            assert response.choices[0].message.content == "Some response"
            # Verify we didn't try to spin up the endpoint
            assert mock_request.call_count == 1

    def test_evaluate_endpoint_needs_spinup(self, mock_chat_response, mock_endpoints_response):
        with patch("modelgauge.suts.together_client._retrying_request") as mock_request:
            # First status check shows STOPPED
            mock_stopped_status = MagicMock()
            mock_stopped_status.json.return_value = {"state": "STOPPED"}

            # Status after PATCH shows STARTED
            mock_started_status = MagicMock()
            mock_started_status.json.return_value = {"state": "STARTED"}

            def side_effect(url, headers, json_payload, method):
                if url.endswith("/endpoints"):
                    return mock_endpoints_response
                elif url.endswith("/test-endpoint-id"):
                    if method == "GET":
                        # First status check returns STOPPED, subsequent ones return STARTED
                        if not hasattr(side_effect, "status_checked"):
                            side_effect.status_checked = True
                            return mock_stopped_status
                        return mock_started_status
                    elif method == "PATCH":
                        return mock_started_status
                elif url == TogetherChatSUT._CHAT_COMPLETIONS_URL:
                    return mock_chat_response
                return MagicMock()

            mock_request.side_effect = side_effect

            # Create SUT and make request
            sut = TogetherDedicatedChatSUT(
                uid="test-model",
                model="some-model",
                api_key=TogetherApiKey("some-value"),
            )
            request = TogetherChatRequest(model="some-model", messages=[])
            response = sut.evaluate(request)

            assert isinstance(response, TogetherChatResponse)
            assert response.choices[0].message.content == "Some response"

    def test_evaluate_400_error_triggers_spinup(self, mock_chat_response, mock_endpoints_response):
        with patch("modelgauge.suts.together_client._retrying_request") as mock_request:
            # Initial status shows STARTED
            mock_status = MagicMock()
            mock_status.json.return_value = {"state": "STARTED"}

            # First chat attempt fails with 400
            mock_400_response = MagicMock()
            mock_400_response.status_code = 400
            mock_400_response.text = "Not Found"

            def raise_400():
                raise HTTPError("400 Not Found")

            mock_400_response.raise_for_status.side_effect = raise_400

            call_count = 0

            def side_effect(url, headers, json_payload, method):
                nonlocal call_count
                if url.endswith("/endpoints"):
                    return mock_endpoints_response
                elif url.endswith("/test-endpoint-id"):
                    return mock_status
                elif url == TogetherChatSUT._CHAT_COMPLETIONS_URL:
                    call_count += 1
                    if call_count == 1:
                        return mock_400_response
                    return mock_chat_response
                return MagicMock()

            mock_request.side_effect = side_effect

            # Create SUT and make request
            sut = TogetherDedicatedChatSUT(
                uid="test-model",
                model="some-model",
                api_key=TogetherApiKey("some-value"),
            )
            request = TogetherChatRequest(model="some-model", messages=[])
            response = sut.evaluate(request)

            assert isinstance(response, TogetherChatResponse)
            assert response.choices[0].message.content == "Some response"
            assert call_count == 2  # Verify we retried after the 400

    def test_evaluate_non_400_error_raises(self, mock_endpoints_response):
        with patch("modelgauge.suts.together_client._retrying_request") as mock_request:
            # Status shows STARTED
            mock_status = MagicMock()
            mock_status.json.return_value = {"state": "STARTED"}

            # Chat attempt fails with 500
            mock_500_response = MagicMock()
            mock_500_response.status_code = 500
            mock_500_response.text = "Internal Server Error"

            def side_effect(url, headers, json_payload, method):
                if url.endswith("/endpoints"):
                    return mock_endpoints_response
                elif url.endswith("/test-endpoint-id"):
                    return mock_status
                elif url == TogetherChatSUT._CHAT_COMPLETIONS_URL:
                    raise APIException("Internal Server Error (500)")
                return MagicMock()

            mock_request.side_effect = side_effect

            # Create SUT and make request
            sut = TogetherDedicatedChatSUT(
                uid="test-model",
                model="some-model",
                api_key=TogetherApiKey("some-value"),
            )
            request = TogetherChatRequest(model="some-model", messages=[])

            # Verify non-400 error is re-raised
            with pytest.raises(APIException, match="Internal Server Error \\(500\\)"):
                sut.evaluate(request)
