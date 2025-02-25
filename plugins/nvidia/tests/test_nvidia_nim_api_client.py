from modelgauge.suts.nvidia_nim_api_client import (
    NvidiaNIMApiKey,
    NvidiaNIMApiClient,
    OpenAIChatMessage,
    OpenAIChatRequest,
)
from openai.types.chat import ChatCompletion

from modelgauge.prompt import TextPrompt
from modelgauge.sut import SUTResponse


def _make_client():
    return NvidiaNIMApiClient(uid="test-model", model="some-model", api_key=NvidiaNIMApiKey("some-value"))


def test_openai_chat_translate_request():
    client = _make_client()
    prompt = TextPrompt(text="some-text")
    request = client.translate_text_prompt(prompt)
    assert request == OpenAIChatRequest(
        model="some-model",
        messages=[OpenAIChatMessage(content="some-text", role="user")],
        max_tokens=100,
        n=1,
    )


def test_openai_chat_translate_response():
    client = _make_client()
    request = OpenAIChatRequest(
        model="some-model",
        messages=[],
    )
    # response is base on openai request: https://platform.openai.com/docs/api-reference/chat/create
    response = ChatCompletion.model_validate_json(
        """\
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "nvidia/nemotron-mini-4b-instruct",
  "system_fingerprint": "fp_44709d6fcb",
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
"""
    )
    result = client.translate_response(request, response)
    assert result == SUTResponse(text="Hello there, how may I assist you today?", top_logprobs=None)
