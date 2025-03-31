import pytest
from unittest.mock import patch

from modelgauge.prompt import TextPrompt
from modelgauge.sut import SUTOptions, SUTResponse
from modelgauge.typed_data import is_typeable

from modelgauge.suts.aws_bedrock_client import (
    AmazonNovaSut,
    AwsAccessKeyId,
    AwsSecretAccessKey,
    BedrockRequest,
    BedrockResponse,
)

FAKE_MODEL_ID = "fake-model"


@pytest.fixture
def fake_sut():
    return AmazonNovaSut(
        "fake-sut", FAKE_MODEL_ID, AwsAccessKeyId("fake-api-key"), AwsSecretAccessKey("fake-secret-key")
    )


def _make_request(model_id, prompt_text, **inference_params):
    inference_config = BedrockRequest.InferenceConfig(**inference_params)
    return BedrockRequest(
        modelId=model_id,
        messages=[
            BedrockRequest.BedrockMessage(content=[{"text": prompt_text}]),
        ],
        inferenceConfig=inference_config,
    )


def _make_response(response_text):
    return BedrockResponse(
        output=BedrockResponse.BedrockResponseOutput(
            message=BedrockResponse.BedrockResponseOutput.BedrockResponseMessage(content=[{"text": response_text}])
        )
    )


def test_translate_text_prompt(fake_sut):
    default_options = SUTOptions()
    prompt = TextPrompt(text="some-text")
    request = fake_sut.translate_text_prompt(prompt, default_options)

    assert isinstance(request, BedrockRequest)
    assert request.modelId == FAKE_MODEL_ID
    assert len(request.messages) == 1
    message = request.messages[0]
    assert message.content == [{"text": "some-text"}]
    assert request.inferenceConfig.maxTokens == default_options.max_tokens  # Default SUTOptions value


def test_can_cache_request():
    request = _make_request(FAKE_MODEL_ID, "some-text", maxTokens=100)
    assert is_typeable(request)


def test_can_cache_response():
    response = _make_response("response")
    assert is_typeable(response)


@patch("modelgauge.suts.aws_bedrock_client.boto3.client")
def test_evaluate_sends_correct_params(mock_client, fake_sut):
    fake_sut.client = mock_client
    request = _make_request(FAKE_MODEL_ID, "some-text", maxTokens=100, topP=0.5)
    fake_sut.evaluate(request)

    mock_client.converse.assert_called_with(
        modelId=FAKE_MODEL_ID,
        messages=[{"content": [{"text": "some-text"}], "role": "user"}],
        inferenceConfig={"maxTokens": 100, "topP": 0.5},
    )


def test_translate_response(fake_sut):
    request = _make_request(FAKE_MODEL_ID, "some-text")
    response = _make_response("response")

    translated_response = fake_sut.translate_response(request, response)

    assert translated_response == SUTResponse(text="response")
