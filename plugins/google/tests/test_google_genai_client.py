import pytest
from unittest.mock import patch

from modelgauge.prompt import SUTOptions, TextPrompt
from modelgauge.suts.google_genai_client import (
    GoogleAiApiKey,
    GoogleGenAiConfig,
    GoogleGenAiSUT,
    GoogleGenAiRequest,
    GoogleGenAiResponse,
)


@pytest.fixture
def google_sut():
    return GoogleGenAiSUT(uid="fake-google-sut", model_name="some-model", api_key=GoogleAiApiKey("some-value"))


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


@patch("modelgauge.suts.google_genai_client.genai.GenerativeModel")
def test_google_genai_evaluate_lazy_loads_model(mock_model, google_sut):
    assert google_sut.model is None

    request = GoogleGenAiRequest(contents="some-text", generation_config=GoogleGenAiConfig())

    google_sut.evaluate(request)

    assert google_sut.model is not None


@patch("modelgauge.suts.google_genai_client.genai.GenerativeModel")
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
