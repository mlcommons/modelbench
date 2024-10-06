import pytest
from google.generativeai.protos import GenerateContentResponse  # type: ignore
from google.generativeai.types import generation_types  # type: ignore

from unittest.mock import patch

from modelgauge.prompt import SUTOptions, TextPrompt
from modelgauge.suts.google_genai_client import (  # type: ignore
    GoogleAiApiKey,
    GoogleGenAiConfig,
    GoogleGenAiSUT,
    GoogleGenAiRequest,
    GoogleGenAiResponse,
)
from modelgauge.sut import SUTCompletion, SUTResponse


@pytest.fixture
def google_sut():
    return GoogleGenAiSUT(uid="fake-google-sut", model_name="some-model", api_key=GoogleAiApiKey("some-value"))


@pytest.fixture
def fake_raw_response():
    response = GenerateContentResponse(
        candidates=[{"content": {"parts": [{"text": "some response"}], "role": "model"}, "finish_reason": 1}],
        usage_metadata={
            "candidates_token_count": 2,
            "prompt_token_count": 0,
            "cached_content_token_count": 0,
            "total_token_count": 0,
        },
    )
    return generation_types.GenerateContentResponse.from_response(response)


@pytest.fixture
def fake_native_response():
    return GoogleGenAiResponse(
        candidates=[
            GoogleGenAiResponse.Candidate(
                content={"parts": [{"text": "some response"}], "role": "model"}, finish_reason=1
            )
        ],
        usage_metadata={
            "candidates_token_count": 2,
            "prompt_token_count": 0,
            "cached_content_token_count": 0,
            "total_token_count": 0,
        },
    )


@pytest.fixture
@patch("modelgauge.suts.google_genai_client.genai.GenerativeModel")
def mock_model(mock_model_patch, fake_raw_response):
    mock_model_patch.generate_content.return_value = fake_raw_response
    return mock_model_patch


def test_google_genai_translate_request_default_options(google_sut):
    prompt = TextPrompt(text="some-text")
    request = google_sut.translate_text_prompt(prompt)
    assert request == GoogleGenAiRequest(
        contents="some-text",
        generation_config=GoogleGenAiConfig(
            stop_sequences=None,
            max_output_tokens=100,
            temperature=None,
            top_p=None,
            top_k=None,
            presence_penalty=None,
            frequency_penalty=None,
        ),
    )


def test_google_genai_translate_request_generation_options(google_sut):
    prompt = TextPrompt(
        text="some-text",
        options=SUTOptions(
            stop_sequences=["stop"], max_tokens=200, temperature=0.5, top_k_per_token=5, frequency_penalty=0.5
        ),
    )
    request = google_sut.translate_text_prompt(prompt)
    assert request == GoogleGenAiRequest(
        contents="some-text",
        generation_config=GoogleGenAiConfig(
            stop_sequences=["stop"],
            max_output_tokens=200,
            temperature=0.5,
            top_p=None,
            top_k=5,
            presence_penalty=None,
            frequency_penalty=0.5,
        ),
    )


@patch("modelgauge.suts.google_genai_client.GoogleGenAiSUT._load_client")
def test_google_genai_evaluate_lazy_loads_model(mock_load_model, mock_model, google_sut):
    mock_load_model.return_value = mock_model
    assert google_sut.model is None

    request = GoogleGenAiRequest(contents="some-text", generation_config=GoogleGenAiConfig())

    google_sut.evaluate(request)

    assert google_sut.model is not None


def test_google_genai_evaluate_excludes_none_options(mock_model, google_sut):
    google_sut.model = mock_model
    request = GoogleGenAiRequest(
        contents="some-text",
        generation_config=GoogleGenAiConfig(stop_sequences=None, max_output_tokens=200, temperature=0.5),
    )

    google_sut.evaluate(request)

    mock_model.generate_content.assert_called_with(
        contents="some-text", generation_config={"max_output_tokens": 200, "temperature": 0.5}
    )


def test_google_genai_evaluate_pydantic_conversion(mock_model, google_sut, fake_native_response):
    google_sut.model = mock_model
    request = GoogleGenAiRequest(contents="some-text", generation_config=GoogleGenAiConfig())

    response = google_sut.evaluate(request)

    assert response == fake_native_response


def test_google_genai_translate_response(google_sut, fake_native_response):
    request = GoogleGenAiRequest(
        contents="some-text",
        generation_config=GoogleGenAiConfig(stop_sequences=None, max_output_tokens=200, temperature=0.5),
    )
    response = google_sut.translate_response(request, fake_native_response)

    assert response == SUTResponse(completions=[SUTCompletion(text="some response")])
