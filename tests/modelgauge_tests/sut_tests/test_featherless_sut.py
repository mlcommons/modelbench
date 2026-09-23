from unittest.mock import MagicMock, patch

import pytest
from openai import OpenAI
from openai._models import construct_type
from openai.types.chat import ChatCompletion

from modelgauge.dynamic_sut_factory import ModelNotSupportedError
from modelgauge.prompt import TextPrompt
from modelgauge.sut import SUTResponse
from modelgauge.sut_definition import SUTDefinition
from modelgauge.retry_decorator import BASE_RETRY_COUNT
from modelgauge.suts.featherless_sut import (
    FEATHERLESS_BASE_URL,
    CapacityError,
    NoOutputError,
    FeatherlessChatRequest,
    FeatherlessSUT,
    FeatherlessSUTFactory,
)
from modelgauge.suts.openai_client import OpenAIChatMessage, OpenAIChatRequest
from modelgauge.model_options import ModelOptions
from modelgauge_tests.utilities import FakeObject


@pytest.fixture
def factory():
    factory = FeatherlessSUTFactory(
        raw_secrets={
            "featherless-ai": {"api_key": "some_key"},
        }
    )
    real_client = OpenAI(api_key="some_key", base_url=FEATHERLESS_BASE_URL)
    list_response = MagicMock()
    list_response.data = [
        FakeObject(id="deepseek-ai/deepseek-v4.1-flash"),
        FakeObject(id="Qwen/Qwen2.5-7B-Instruct"),
    ]
    real_client.models.list = MagicMock(return_value=list_response)
    factory._client = real_client
    return factory


def test_factory_uses_api_model_id_casing(factory):
    factory.client.models.list.return_value.data = [
        FakeObject(id="deepseek-ai/DeepSeek-V4.1-Flash"),
    ]
    sut_definition = SUTDefinition(
        maker="deepseek-ai",
        model="deepseek-v4.1-flash",
        driver="featherless-ai",
    )
    sut = factory.make_sut(sut_definition)
    assert sut.model == "deepseek-ai/DeepSeek-V4.1-Flash"


def test_factory_makes_correct_featherless_sut(factory):
    sut_definition = SUTDefinition(
        maker="deepseek-ai",
        model="DeepSeek-V4.1-Flash",
        driver="featherless-ai",
    )
    sut = factory.make_sut(sut_definition)

    assert isinstance(sut, FeatherlessSUT)
    assert sut.uid == "deepseek-ai/deepseek-v4.1-flash:featherless-ai"
    # Featherless model IDs are case-sensitive; use the casing from their /models list.
    assert sut.model == "deepseek-ai/deepseek-v4.1-flash"
    assert sut.client is factory.client
    assert str(factory.client.base_url).startswith(FEATHERLESS_BASE_URL)
    factory.client.models.list.assert_called()


def test_make_sut_bad_model(factory):
    sut_definition = SUTDefinition(
        maker="deepseek-ai",
        model="bogus",
        driver="featherless-ai",
    )
    with pytest.raises(ModelNotSupportedError):
        factory.make_sut(sut_definition)


def test_list_suts(factory):
    suts = factory.list_suts()
    assert suts is not None
    uids = [s.uid for s in suts]
    assert "deepseek-ai/deepseek-v4.1-flash:featherless-ai" in uids
    assert "qwen/qwen2.5-7b-instruct:featherless-ai" in uids


def _make_sut():
    return FeatherlessSUT(uid="test-model", model="some-model", api_key=None, client=MagicMock())


def test_translate_text_prompt_uses_chat_completions():
    sut = _make_sut()
    prompt = TextPrompt(text="some-text")
    request = sut.translate_text_prompt(prompt, ModelOptions(max_tokens=100, temperature=0.01))
    assert request == FeatherlessChatRequest(
        model="some-model",
        messages=[OpenAIChatMessage(content="some-text", role="user")],
        max_tokens=100,
        temperature=0.01,
    )
    payload = sut.request_as_dict_for_client(request)
    assert payload["max_tokens"] == 100
    assert "max_completion_tokens" not in payload


def test_translate_response():
    sut = _make_sut()
    request = OpenAIChatRequest(
        model="some-model",
        messages=[],
    )
    response = ChatCompletion.model_validate_json("""\
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "some-model",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello there, how may I assist you today?"
    },
    "logprobs": null,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 12,
    "total_tokens": 21
  }
}
""")
    result = sut.translate_response(request, response)
    assert result == SUTResponse(text="Hello there, how may I assist you today?", top_logprobs=None)


def _capacity_response():
    return construct_type(
        type_=ChatCompletion,
        value={
            "error": {
                "message": "some-model is temporarily at capacity. Please try again shortly.",
                "type": "server_error",
                "code": "capacity_exhausted",
            }
        },
    )


def _completion(content="ok"):
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "some-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
    )


def test_evaluate_retries_capacity_errors_until_success():
    sut = _make_sut()
    success = _completion()
    sut.client.chat.completions.create = MagicMock(side_effect=[_capacity_response(), _capacity_response(), success])
    request = sut.translate_text_prompt(TextPrompt(text="some-text"), ModelOptions(max_tokens=20))

    with patch("time.sleep") as sleep:
        result = sut.evaluate(request)

    assert result is success
    assert sut.client.chat.completions.create.call_count == 3
    assert sleep.call_count == 2


def _no_output_response():
    return construct_type(
        type_=ChatCompletion,
        value={
            "error": {
                "message": "The model produced no output. Please try again.",
                "type": "server_error",
                "code": "no_output",
            }
        },
    )


def test_evaluate_retries_no_output_a_limited_number_of_times():
    sut = _make_sut()
    sut.client.chat.completions.create = MagicMock(return_value=_no_output_response())
    request = sut.translate_text_prompt(TextPrompt(text="some-text"), ModelOptions(max_tokens=20))

    with patch("time.sleep"):
        with pytest.raises(NoOutputError, match="The model produced no output"):
            sut.evaluate(request)

    assert sut.client.chat.completions.create.call_count == BASE_RETRY_COUNT


def test_call_client_raises_no_output_error_without_retrying():
    sut = _make_sut()
    sut.client.chat.completions.create = MagicMock(return_value=_no_output_response())
    request = sut.translate_text_prompt(TextPrompt(text="some-text"), ModelOptions(max_tokens=20))

    with pytest.raises(NoOutputError):
        sut._call_client(request)

    assert sut.client.chat.completions.create.call_count == 1


def test_call_client_raises_capacity_error_without_retrying():
    sut = _make_sut()
    sut.client.chat.completions.create = MagicMock(return_value=_capacity_response())
    request = sut.translate_text_prompt(TextPrompt(text="some-text"), ModelOptions(max_tokens=20))

    with pytest.raises(CapacityError):
        sut._call_client(request)

    assert sut.client.chat.completions.create.call_count == 1


def test_evaluate_stops_retrying_other_errors():
    sut = _make_sut()
    sut.client.chat.completions.create = MagicMock(side_effect=ValueError("bad request"))
    request = sut.translate_text_prompt(TextPrompt(text="some-text"), ModelOptions(max_tokens=20))

    with patch("time.sleep"):
        with pytest.raises(ValueError, match="bad request"):
            sut.evaluate(request)

    assert sut.client.chat.completions.create.call_count == BASE_RETRY_COUNT
