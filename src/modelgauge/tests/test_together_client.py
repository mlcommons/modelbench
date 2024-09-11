from unittest.mock import patch, MagicMock

import pytest
from requests import HTTPError  # type:ignore

from modelgauge.general import APIException
from modelgauge.prompt import SUTOptions, ChatMessage, ChatPrompt, ChatRole, TextPrompt
from modelgauge.prompt_formatting import format_chat
from modelgauge.sut import SUTCompletion, SUTResponse, TokenProbability, TopTokens
from modelgauge.suts.together_client import (
    TogetherApiKey,
    TogetherChatResponse,
    TogetherChatRequest,
    TogetherChatSUT,
    TogetherCompletionsResponse,
    TogetherCompletionsRequest,
    TogetherCompletionsSUT,
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
    request = client.translate_text_prompt(prompt)
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
    request = client.translate_chat_prompt(prompt)
    assert request == request_class(
        model="some-model",
        prompt=format_chat(prompt, user_role="user", sut_role="assistant"),
        max_tokens=100,
        n=1,
    )


def test_together_chat_translate_text_prompt_request():
    client = _make_client(TogetherChatSUT)
    prompt = TextPrompt(text="some-text")
    request = client.translate_text_prompt(prompt)
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
    request = client.translate_chat_prompt(prompt)
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
    prompt = TextPrompt(text="some-text", options=SUTOptions(top_logprobs=1))
    request = client.translate_text_prompt(prompt)
    assert request == request_class(
        model="some-model",
        prompt="some-text",
        max_tokens=100,
        n=1,
        logprobs=1,
    )


def test_together_chat_translate_request_logprobs():
    client = _make_client(TogetherChatSUT)
    prompt = TextPrompt(text="some-text", options=SUTOptions(top_logprobs=1))
    request = client.translate_text_prompt(prompt)
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
    assert result == SUTResponse(
        completions=[SUTCompletion(text=" blue.", top_logprobs=None)]
    )


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
        completions=[
            SUTCompletion(
                text=" blue.",
                top_logprobs=[
                    TopTokens(
                        top_tokens=[TokenProbability(token=" blue", logprob=-1.9072266)]
                    ),
                    TopTokens(
                        top_tokens=[TokenProbability(token=".", logprob=-0.703125)]
                    ),
                ],
            )
        ]
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
    assert result == SUTResponse(
        completions=[SUTCompletion(text=" blue.", top_logprobs=None)]
    )


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
        completions=[
            SUTCompletion(
                text=" blue.",
                top_logprobs=[
                    TopTokens(
                        top_tokens=[TokenProbability(token=" blue", logprob=-1.9072266)]
                    ),
                    TopTokens(
                        top_tokens=[TokenProbability(token=".", logprob=-0.703125)]
                    ),
                ],
            )
        ]
    )


def test_together_chat_evaluate_normal():
    client = _make_client(TogetherChatSUT)
    with patch("modelgauge.suts.together_client._retrying_post") as mock_post:
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
    assert result == SUTResponse(
        completions=[SUTCompletion(text="Some response", top_logprobs=None)]
    )


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
        completions=[
            SUTCompletion(
                text="Some response",
                top_logprobs=[
                    TopTokens(
                        top_tokens=[TokenProbability(token="Some", logprob=-0.55810547)]
                    ),
                    TopTokens(
                        top_tokens=[
                            TokenProbability(token="response", logprob=-0.9326172)
                        ]
                    ),
                ],
            )
        ]
    )
