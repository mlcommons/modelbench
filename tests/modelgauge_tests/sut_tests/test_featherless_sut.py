from unittest.mock import MagicMock

import pytest
from openai import OpenAI
from openai.types.chat import ChatCompletion

from modelgauge.dynamic_sut_factory import ModelNotSupportedError
from modelgauge.prompt import TextPrompt
from modelgauge.sut import SUTResponse
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.featherless_sut import (
    FEATHERLESS_BASE_URL,
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
            "featherless": {"api_key": "some_key"},
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
        driver="featherless",
    )
    sut = factory.make_sut(sut_definition)
    assert sut.model == "deepseek-ai/DeepSeek-V4.1-Flash"


def test_factory_makes_correct_featherless_sut(factory):
    sut_definition = SUTDefinition(
        maker="deepseek-ai",
        model="DeepSeek-V4.1-Flash",
        driver="featherless",
    )
    sut = factory.make_sut(sut_definition)

    assert isinstance(sut, FeatherlessSUT)
    assert sut.uid == "deepseek-ai/deepseek-v4.1-flash:featherless"
    # Featherless model IDs are case-sensitive; use the casing from their /models list.
    assert sut.model == "deepseek-ai/deepseek-v4.1-flash"
    assert sut.client is factory.client
    assert str(factory.client.base_url).startswith(FEATHERLESS_BASE_URL)
    factory.client.models.list.assert_called()


def test_make_sut_bad_model(factory):
    sut_definition = SUTDefinition(
        maker="deepseek-ai",
        model="bogus",
        driver="featherless",
    )
    with pytest.raises(ModelNotSupportedError):
        factory.make_sut(sut_definition)


def test_list_suts(factory):
    suts = factory.list_suts()
    assert suts is not None
    uids = [s.uid for s in suts]
    assert "deepseek-ai/deepseek-v4.1-flash:featherless" in uids
    assert "qwen/qwen2.5-7b-instruct:featherless" in uids


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
