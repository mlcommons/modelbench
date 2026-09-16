import json
from unittest.mock import MagicMock

import pytest
from google.genai.types import (
    FinishReason,
    GenerateContentConfig,
    GenerateContentResponse,
    ThinkingConfig,
    ThinkingLevel,
)

from modelgauge.general import APIException
from modelgauge.model_options import ModelOptions
from modelgauge.prompt import TextPrompt
from modelgauge.sut import REFUSAL_RESPONSE, SUTResponse
from modelgauge.suts.google_genai import GenAiRequest, GoogleGenAiSUT

_MODEL_NAME = "some-model"


@pytest.fixture
def google_default_sut():
    return GoogleGenAiSUT(uid="fake-google-sut", model_name=_MODEL_NAME, use_reasoning=True, client=MagicMock())


@pytest.fixture
def google_unreasoning_sut():
    return GoogleGenAiSUT(uid="fake-google-sut", model_name=_MODEL_NAME, use_reasoning=False, client=MagicMock())


@pytest.fixture
def some_request():
    return GenAiRequest(model=_MODEL_NAME, contents="some-text", config=GenerateContentConfig())


@pytest.fixture
def fake_raw_response():
    response = GenerateContentResponse(
        candidates=[{"content": {"parts": [{"text": "some response"}], "role": "model"}}],
        usage_metadata={
            "candidates_token_count": 2,
            "prompt_token_count": 0,
            "cached_content_token_count": 0,
            "total_token_count": 2,
        },
    )
    return response


def test_google_genai_translate_request_default_options(google_default_sut):
    prompt = TextPrompt(text="some-text")
    request = google_default_sut.translate_text_prompt(prompt, ModelOptions())
    assert request == GenAiRequest(
        model=_MODEL_NAME,
        contents="some-text",
        config=GenerateContentConfig(
            stop_sequences=None,
            max_output_tokens=None,
            temperature=None,
            top_p=None,
            top_k=None,
            presence_penalty=None,
            frequency_penalty=None,
        ),
    )


def test_google_genai_translate_request_default_options_no_reasoning(google_unreasoning_sut):
    prompt = TextPrompt(text="some-text")
    request = google_unreasoning_sut.translate_text_prompt(prompt, ModelOptions())
    assert request == GenAiRequest(
        model=_MODEL_NAME,
        contents="some-text",
        config=GenerateContentConfig(
            stop_sequences=None,
            max_output_tokens=None,
            temperature=None,
            top_p=None,
            top_k=None,
            presence_penalty=None,
            frequency_penalty=None,
            thinking_config=ThinkingConfig(thinking_budget=0),
        ),
    )


@pytest.mark.parametrize("model_name", ["gemma-4-26b-a4b-it", "gemini-3-flash"])
def test_google_genai_translate_request_no_reasoning_uses_thinking_level(model_name):
    sut = GoogleGenAiSUT(uid="fake-google-sut", model_name=model_name, use_reasoning=False, client=MagicMock())
    prompt = TextPrompt(text="some-text")
    request = sut.translate_text_prompt(prompt, ModelOptions())
    assert request.config.thinking_config == ThinkingConfig(thinking_level=ThinkingLevel.MINIMAL)


def test_google_genai_translate_request_generation_options(google_default_sut):
    prompt = TextPrompt(text="some-text")
    options = ModelOptions(
        stop_sequences=["stop"], max_tokens=200, temperature=0.5, top_k_per_token=5, frequency_penalty=0.5
    )
    request = google_default_sut.translate_text_prompt(prompt, options)
    assert request == GenAiRequest(
        model=_MODEL_NAME,
        contents="some-text",
        config=GenerateContentConfig(
            stop_sequences=["stop"],
            max_output_tokens=200,
            temperature=0.5,
            top_p=None,
            top_k=5,
            presence_penalty=None,
            frequency_penalty=0.5,
        ),
    )


def test_google_genai_evaluate_excludes_none_options(google_default_sut):
    request = GenAiRequest(
        model=_MODEL_NAME,
        contents="some-text",
        config=GenerateContentConfig(stop_sequences=None, max_output_tokens=200, temperature=0.5),
    )

    google_default_sut.evaluate(request)

    google_default_sut.client.models.generate_content.assert_called_with(
        model=_MODEL_NAME, contents="some-text", config={"max_output_tokens": 200, "temperature": 0.5}
    )


def test_google_genai_evaluate_unreasoning(google_unreasoning_sut):
    request = GenAiRequest(
        model=_MODEL_NAME,
        contents="some-text",
        config=GenerateContentConfig(thinking_config=ThinkingConfig(thinking_budget=0)),
    )

    google_unreasoning_sut.evaluate(request)

    google_unreasoning_sut.client.models.generate_content.assert_called_with(
        model=_MODEL_NAME, contents="some-text", config={"thinking_config": {"thinking_budget": 0}}
    )


def test_google_genai_translate_response(google_default_sut, fake_raw_response, some_request):
    response = google_default_sut.translate_response(some_request, fake_raw_response)

    assert response == SUTResponse(text="some response")


def test_google_genai_translate_response_uses_only_non_thinking_part(google_default_sut, some_request):
    raw_response = GenerateContentResponse(
        candidates=[
            {
                "content": {
                    "parts": [
                        {"text": "thinking text", "thought": True},
                        {"text": "visible response"},
                        {"text": "more thinking text", "thought": True},
                    ],
                    "role": "model",
                }
            }
        ],
    )

    response = google_default_sut.translate_response(some_request, raw_response)

    assert response == SUTResponse(text="visible response")


@pytest.mark.parametrize(
    "parts",
    [
        [{"text": "thinking text", "thought": True}],
        [],
        [{"text": "part 1"}, {"text": "part 2"}],
    ],
)
def test_google_genai_translate_response_raises_without_one_non_thinking_part(google_default_sut, some_request, parts):
    raw_response = GenerateContentResponse(
        candidates=[{"content": {"parts": parts, "role": "model"}}],
    )

    with pytest.raises(APIException, match="Expected 1 non-thinking part"):
        google_default_sut.translate_response(some_request, raw_response)


def test_google_genai_translate_response_finish_reason_other(google_default_sut, fake_raw_response, some_request):
    """I think this is for a typing error but we're in a rush so I'm not fixing it"""
    fake_raw_response.candidates[0].finish_reason = FinishReason.OTHER
    response = google_default_sut.translate_response(some_request, fake_raw_response)

    assert response == SUTResponse(text="")  # indicates refusal


def test_google_genai_translate_response_no_completions(google_default_sut, some_request):
    no_completions = GenerateContentResponse(**json.loads("""{
  "candidates": [],
  "usage_metadata": {
    "prompt_token_count": 19,
    "total_token_count": 19,
    "cached_content_token_count": 0,
    "candidates_token_count": 0
  }
}
"""))
    response = google_default_sut.translate_response(some_request, no_completions)

    assert response == SUTResponse(text=REFUSAL_RESPONSE)


def test_google_genai_translate_response_none_completions(google_default_sut, some_request):
    no_completions = GenerateContentResponse(**json.loads("""{
  "candidates": null,
  "usage_metadata": {
    "prompt_token_count": 19,
    "total_token_count": 19,
    "cached_content_token_count": 0,
    "candidates_token_count": 0
  }
}
"""))
    response = google_default_sut.translate_response(some_request, no_completions)

    assert response == SUTResponse(text=REFUSAL_RESPONSE)
