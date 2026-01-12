from unittest.mock import MagicMock

from llama_api_client.types import CreateChatCompletionResponse

from modelgauge.prompt import TextPrompt
from modelgauge.sut import SUTResponse
from modelgauge.model_options import ModelOptions
from modelgauge.suts.meta_llama_client import InputMessage, MetaLlamaApiKey, MetaLlamaChatRequest, MetaLlamaSUT
from pytest import fixture
from requests import HTTPError  # type:ignore

llama_chat_response_text = """
{
  "completion_message": {
    "role": "assistant",
    "stop_reason": "stop",
    "content": {
      "type": "text",
      "text": "The classic joke! There are many possible answers, but the most common one is: \\"To get to the other side!\\" Would you like to hear some variations or alternative punchlines?"
    }
  },
  "metrics": [
    {
      "metric": "num_completion_tokens",
      "value": 38,
      "unit": "tokens"
    },
    {
      "metric": "num_prompt_tokens",
      "value": 22,
      "unit": "tokens"
    },
    {
      "metric": "num_total_tokens",
      "value": 60,
      "unit": "tokens"
    }
  ]
}
"""


@fixture
def sut():
    return MetaLlamaSUT("ignored", "a_model", MetaLlamaApiKey("whatever"))


def test_translate_text_prompt(sut):
    sut_options = ModelOptions()
    result = sut.translate_text_prompt(TextPrompt(text="Why did the chicken cross the road?"), sut_options)
    assert result == MetaLlamaChatRequest(
        model="a_model",
        messages=[InputMessage(role="user", content="Why did the chicken cross the road?")],
        max_completion_tokens=sut_options.max_tokens,
    )


def test_translate_chat_response(sut):
    request = MetaLlamaChatRequest(
        model="a_model",
        messages=[InputMessage(role="user", content="Why did the chicken cross the road?")],
    )
    response = CreateChatCompletionResponse.model_validate_json(llama_chat_response_text)
    result = sut.translate_response(request, response)
    assert result == SUTResponse(
        text='The classic joke! There are many possible answers, but the most common one is: "To get to the other side!" Would you like to hear some variations or alternative punchlines?'
    )


def test_evaluate(sut):
    request = MetaLlamaChatRequest(
        model="a_model",
        messages=[InputMessage(role="user", content="Why did the chicken cross the road?")],
        max_completion_tokens=123,
    )
    sut.client = MagicMock()
    _ = sut.evaluate(request)
    assert sut.client.chat.completions.create.call_count == 1
    kwargs = sut.client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "a_model"
    assert kwargs["messages"][0]["role"] == "user"
    assert kwargs["messages"][0]["content"] == "Why did the chicken cross the road?"
    assert kwargs["max_completion_tokens"] == 123
    assert "temperature" not in kwargs
