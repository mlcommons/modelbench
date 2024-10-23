import pytest
from unittest.mock import ANY, patch

from modelgauge.auth.huggingface_inference_token import HuggingFaceInferenceToken
from modelgauge.prompt import SUTOptions, TextPrompt
from modelgauge.sut import SUTCompletion, SUTResponse
from modelgauge.suts.huggingface_api import (
    HuggingFaceChatParams,
    HuggingFaceChatRequest,
    HuggingFaceResponse,
    HuggingFaceSUT,
)


@pytest.fixture
def fake_sut():
    return HuggingFaceSUT("fake_uid", "https://fake_url.com", HuggingFaceInferenceToken("fake_token"))


@pytest.fixture
def prompt():
    return TextPrompt(
        text="some text prompt",
        options=SUTOptions(max_tokens=5, temperature=1.0, random="random"),
    )


@pytest.fixture
def sut_request():
    return HuggingFaceChatRequest(
        inputs="some text prompt", parameters=HuggingFaceChatParams(max_new_tokens=5, temperature=1.0)
    )


def test_huggingface_api_translate_text_prompt_request(fake_sut, prompt, sut_request):
    request = fake_sut.translate_text_prompt(prompt)

    assert isinstance(request, HuggingFaceChatRequest)
    assert request == sut_request


def mocked_requests_post():
    class MockResponse:
        def __init__(self, json_data, status_code):
            self.json_data = json_data
            self.status_code = status_code

        def json(self):
            return [self.json_data]

    return MockResponse({"generated_text": "response"}, 200)


@patch("requests.post", side_effect=mocked_requests_post)
def test_huggingface_api_evaluate_receives_correct_args(mock_post, fake_sut, sut_request):
    fake_sut.evaluate(sut_request)

    mock_post.assert_called_with(
        "https://fake_url.com",
        headers=ANY,
        json={"inputs": "some text prompt", "parameters": {"max_new_tokens": 5, "temperature": 1.0}},
    )


@patch("requests.post", side_effect=mocked_requests_post)
def test_huggingface_api_evaluate_dumps_result(mock_post, fake_sut, sut_request):
    output = fake_sut.evaluate(sut_request)

    assert output == HuggingFaceResponse(generated_text="response")


def test_huggingface_chat_completion_translate_response(fake_sut, sut_request):
    evaluate_output = HuggingFaceResponse(generated_text="response")

    response = fake_sut.translate_response(sut_request, evaluate_output)

    assert response == SUTResponse(completions=[SUTCompletion(text="response")])
