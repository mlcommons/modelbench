import pytest
from unittest.mock import ANY, patch

from modelgauge.auth.huggingface_inference_token import HuggingFaceInferenceToken
from modelgauge.prompt import TextPrompt
from modelgauge.sut import SUTOptions, SUTResponse
from modelgauge.suts.huggingface_api import (
    HuggingFaceChatParams,
    HuggingFaceChatRequest,
    HuggingFaceResponse,
    HuggingFaceSUT,
)


@pytest.fixture
def fake_sut():
    return HuggingFaceSUT("fake_uid", "https://fake_url.com", HuggingFaceInferenceToken("fake_token"))


def _make_sut_request(text, **params):
    return HuggingFaceChatRequest(inputs=text, parameters=HuggingFaceChatParams(**params))


def test_huggingface_api_translate_text_prompt_request(fake_sut):
    prompt_text = "some text prompt"
    sut_options = SUTOptions(max_tokens=5, temperature=1.0, random="should be ignored")
    prompt = TextPrompt(text=prompt_text)

    request = fake_sut.translate_text_prompt(prompt, sut_options)

    assert isinstance(request, HuggingFaceChatRequest)
    assert request.inputs == prompt_text
    assert request.parameters == HuggingFaceChatParams(max_new_tokens=5, temperature=1.0)


def mocked_requests_post(response_text):
    class MockResponse:
        def __init__(self, json_data, status_code):
            self.json_data = json_data
            self.status_code = status_code

        def json(self):
            return [self.json_data]

    return MockResponse({"generated_text": response_text}, 200)


@patch("requests.post")
def test_huggingface_api_evaluate_receives_correct_args(mock_post, fake_sut):
    mock_post.return_value = mocked_requests_post("doesn't matter")
    prompt_text = "some text prompt"
    sut_options = {"max_new_tokens": 5, "temperature": 1.0}
    sut_request = _make_sut_request(prompt_text, **sut_options)

    fake_sut.evaluate(sut_request)

    mock_post.assert_called_with(
        "https://fake_url.com",
        headers=ANY,
        json={"inputs": prompt_text, "parameters": sut_options},
    )


@patch("requests.post")
def test_huggingface_api_evaluate_dumps_result(mock_post, fake_sut):
    response_text = "some response"
    mock_post.return_value = mocked_requests_post(response_text)

    output = fake_sut.evaluate(_make_sut_request("some text prompt"))

    assert output == HuggingFaceResponse(generated_text=response_text)


def test_huggingface_chat_completion_translate_response(fake_sut):
    sut_request = _make_sut_request("doesn't matter")
    evaluate_output = HuggingFaceResponse(generated_text="response")

    response = fake_sut.translate_response(sut_request, evaluate_output)

    assert response == SUTResponse(text="response")
