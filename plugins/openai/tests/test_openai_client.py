from modelgauge.prompt import SUTOptions, TextPrompt
from modelgauge.sut import SUTCompletion, SUTResponse, TokenProbability, TopTokens
from modelgauge.suts.openai_client import (
    OpenAIApiKey,
    OpenAIChat,
    OpenAIChatMessage,
    OpenAIChatRequest,
    OpenAIOrgId,
)
from openai.types.chat import ChatCompletion


def _make_client():
    return OpenAIChat(
        uid="test-model",
        model="some-model",
        api_key=OpenAIApiKey("some-value"),
        org_id=OpenAIOrgId(None),
    )


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


def test_openai_chat_translate_request_logprobs():
    client = _make_client()
    prompt = TextPrompt(text="some-text", options=SUTOptions(top_logprobs=2))
    request = client.translate_text_prompt(prompt)
    assert request == OpenAIChatRequest(
        model="some-model",
        messages=[OpenAIChatMessage(content="some-text", role="user")],
        max_tokens=100,
        n=1,
        logprobs=True,
        top_logprobs=2,
    )


def test_openai_chat_translate_request_excessive_logprobs():
    client = _make_client()
    # Set value above limit of 20
    prompt = TextPrompt(text="some-text", options=SUTOptions(top_logprobs=21))
    request = client.translate_text_prompt(prompt)
    assert request == OpenAIChatRequest(
        model="some-model",
        messages=[OpenAIChatMessage(content="some-text", role="user")],
        max_tokens=100,
        n=1,
        logprobs=True,
        top_logprobs=20,
    )


def test_openai_chat_translate_response():
    client = _make_client()
    request = OpenAIChatRequest(
        model="some-model",
        messages=[],
    )
    # Pulled from https://platform.openai.com/docs/api-reference/chat/create
    response = ChatCompletion.model_validate_json(
        """\
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-3.5-turbo-0125",
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
    assert result == SUTResponse(
        completions=[
            SUTCompletion(
                text="Hello there, how may I assist you today?", top_logprobs=None
            )
        ]
    )


def test_openai_chat_translate_response_logprobs():
    client = _make_client()
    request = OpenAIChatRequest(
        model="some-model",
        messages=[],
        logprobs=True,
    )
    # Copied from a real response.
    response = ChatCompletion.model_validate_json(
        """\
{
  "id": "made-this-fake",
  "choices": [
    {
      "finish_reason": "length",
      "index": 0,
      "logprobs": {
        "content": [
          {
            "token": "Hello",
            "logprob": -0.10257129,
            "top_logprobs": [
              {
                "token": "Hello",
                "bytes": [
                  72,
                  101,
                  108,
                  108,
                  111
                ],
                "logprob": -0.10257129
              },
              {
                "token": "Hi",
                "bytes": [
                  72,
                  105
                ],
                "logprob": -2.349693
              }
            ]
          },
          {
            "token": "!",
            "bytes": [
              33
            ],
            "logprob": -0.009831643,
            "top_logprobs": [
              {
                "token": "!",
                "bytes": [
                  33
                ],
                "logprob": -0.009831643
              },
              {
                "token": " there",
                "bytes": [
                  32,
                  116,
                  104,
                  101,
                  114,
                  101
                ],
                "logprob": -4.699771
              }
            ]
          }
        ]
      },
      "message": {
        "content": "Hello!",
        "role": "assistant",
        "function_call": null,
        "tool_calls": null
      }
    }
  ],
  "created": 1711044293,
  "model": "gpt-3.5-turbo-0125",
  "object": "chat.completion",
  "system_fingerprint": "fp_fa89f7a861",
  "usage": {
    "completion_tokens": 2,
    "prompt_tokens": 9,
    "total_tokens": 11
  }
}
"""
    )
    result = client.translate_response(request, response)
    assert result == SUTResponse(
        completions=[
            SUTCompletion(
                text="Hello!",
                top_logprobs=[
                    TopTokens(
                        top_tokens=[
                            TokenProbability(token="Hello", logprob=-0.10257129),
                            TokenProbability(token="Hi", logprob=-2.349693),
                        ]
                    ),
                    TopTokens(
                        top_tokens=[
                            TokenProbability(token="!", logprob=-0.009831643),
                            TokenProbability(token=" there", logprob=-4.699771),
                        ]
                    ),
                ],
            )
        ]
    )
