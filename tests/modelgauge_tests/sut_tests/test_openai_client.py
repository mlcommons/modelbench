import pytest

from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.responses import Response

from modelgauge.prompt import TextPrompt
from modelgauge.sut import SUTResponse
from modelgauge.model_options import ModelOptions, TokenProbability, TopTokens
from modelgauge.suts.openai_client import (
    BaseOpenAI,
    OpenAIApiKey,
    OpenAIChat,
    OpenAIChatMessage,
    OpenAIChatRequest,
    OpenAIOrganization,
    OpenAIResponses,
    OpenAIResponsesRequest,
)


@pytest.fixture
def openai_client():
    return OpenAI(api_key="some-value", organization="some-org", max_retries=1)


class TestBaseOpenAIEvaluate:

    def test_check_accepts_temp(self):
        assert BaseOpenAI._check_accepts_temperature("some-model") is True
        assert BaseOpenAI._check_accepts_temperature("gpt-4o") is True
        assert BaseOpenAI._check_accepts_temperature("gpt-4.6") is True
        assert BaseOpenAI._check_accepts_temperature("gpt-5.4") is True
        assert BaseOpenAI._check_accepts_temperature("gpt-5.4-nano") is True
        assert BaseOpenAI._check_accepts_temperature("gpt-5.4-mini") is True

        assert BaseOpenAI._check_accepts_temperature("gpt-5.6") is False
        assert BaseOpenAI._check_accepts_temperature("gpt-5.6-sol") is False
        assert BaseOpenAI._check_accepts_temperature("gpt-5.5") is False
        assert BaseOpenAI._check_accepts_temperature("gpt-5.5-pro") is False
        assert BaseOpenAI._check_accepts_temperature("gpt-5.4-pro") is False
        assert BaseOpenAI._check_accepts_temperature("gpt-5.7") is False
        assert BaseOpenAI._check_accepts_temperature("gpt-5.10") is False
        assert BaseOpenAI._check_accepts_temperature("gpt-5.10.0") is False


class TestOpenAIChat:
    @pytest.fixture
    def client(self):
        return OpenAIChat(
            uid="test-model",
            model="some-model",
            api_key=OpenAIApiKey("some-value"),
            organization=OpenAIOrganization(None),
        )

    def test_openai_constructor(self, openai_client):
        # these should all work
        key_only = OpenAIChat(
            uid="test-model",
            model="some-model",
            api_key=OpenAIApiKey("some-value"),
            organization=OpenAIOrganization(None),
        )
        key_and_org = OpenAIChat(
            uid="test-model",
            model="some-model",
            api_key=OpenAIApiKey("some-value"),
            organization=OpenAIOrganization("some-org"),
        )
        key_and_base_url = OpenAIChat(
            uid="test-model",
            model="some-model",
            api_key=OpenAIApiKey("some-value"),
            base_url="some-url",
        )

        with_client = OpenAIChat(
            uid="test-model",
            model="some-model",
            client=openai_client,  # type: ignore
        )

        # these should all fail

        # no key and no client
        with pytest.raises(AssertionError):
            _ = OpenAIChat(uid="test-model", model="some-model")

        # base_url and organization
        with pytest.raises(AssertionError):
            _ = OpenAIChat(
                uid="test-model",
                model="some-model",
                organization=OpenAIOrganization("some-org"),
                base_url="some-url",
            )

    def test_openai_chat_translate_request(self, client):
        prompt = TextPrompt(text="some-text")
        request = client.translate_text_prompt(prompt, ModelOptions(max_tokens=100))
        assert request == OpenAIChatRequest(
            model="some-model",
            messages=[OpenAIChatMessage(content="some-text", role="user")],
            max_completion_tokens=100,
        )

    def test_openai_chat_translate_request_logprobs(self, client):
        prompt = TextPrompt(text="some-text")
        request = client.translate_text_prompt(prompt, ModelOptions(top_logprobs=2))
        assert request == OpenAIChatRequest(
            model="some-model",
            messages=[OpenAIChatMessage(content="some-text", role="user")],
            max_completion_tokens=None,
            logprobs=True,
            top_logprobs=2,
        )

    def test_openai_chat_translate_request_excessive_logprobs(self, client):
        # Set value above limit of 20
        prompt = TextPrompt(text="some-text")
        request = client.translate_text_prompt(prompt, ModelOptions(top_logprobs=21))
        assert request == OpenAIChatRequest(
            model="some-model",
            messages=[OpenAIChatMessage(content="some-text", role="user")],
            max_completion_tokens=None,
            logprobs=True,
            top_logprobs=20,
        )

    def test_openai_chat_translate_response(self, client):
        request = OpenAIChatRequest(
            model="some-model",
            messages=[],
        )
        # Pulled from https://platform.openai.com/docs/api-reference/chat/create
        response = ChatCompletion.model_validate_json("""\
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
          """)
        result = client.translate_response(request, response)
        assert result == SUTResponse(text="Hello there, how may I assist you today?", top_logprobs=None)

    def test_openai_chat_translate_response_logprobs(self, client):
        request = OpenAIChatRequest(
            model="some-model",
            messages=[],
            logprobs=True,
        )
        # Copied from a real response.
        response = ChatCompletion.model_validate_json("""\
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
        """)
        result = client.translate_response(request, response)
        assert result == SUTResponse(
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

    def test_translate_request_includes_temperature_when_accepted(self, client):
        """When _accepts_temperature is True, temperature from options is included in the request."""
        assert client._accepts_temperature is True
        prompt = TextPrompt(text="some-text")
        request = client.translate_text_prompt(prompt, ModelOptions(temperature=0.7))
        assert request.temperature == 0.7

    def test_translate_request_omits_temperature_when_not_accepted(self, client):
        """When _accepts_temperature is False, temperature is excluded from the request even if provided."""
        client._accepts_temperature = False
        prompt = TextPrompt(text="some-text")
        request = client.translate_text_prompt(prompt, ModelOptions(temperature=0.7))
        assert request.temperature is None


def _make_response(text: str, logprobs=None) -> Response:
    content = {"type": "output_text", "text": text, "annotations": []}
    if logprobs is not None:
        content["logprobs"] = logprobs
    return Response.model_validate(
        {
            "id": "resp-fake",
            "object": "response",
            "created_at": 1677652288,
            "model": "gpt-4o",
            "status": "completed",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "type": "message",
                    "id": "msg-fake",
                    "role": "assistant",
                    "status": "completed",
                    "content": [content],
                }
            ],
            "usage": {
                "input_tokens": 9,
                "output_tokens": 12,
                "total_tokens": 21,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
    )


class TestOpenAIResponses:
    @pytest.fixture
    def client(self):
        return OpenAIResponses(
            uid="test-model",
            model="some-model",
            api_key=OpenAIApiKey("some-value"),
            organization=OpenAIOrganization(None),
        )

    def test_openai_constructor(self, openai_client):
        # these should all work
        key_only = OpenAIResponses(
            uid="test-model",
            model="some-model",
            api_key=OpenAIApiKey("some-value"),
            organization=OpenAIOrganization(None),
        )
        key_and_org = OpenAIResponses(
            uid="test-model",
            model="some-model",
            api_key=OpenAIApiKey("some-value"),
            organization=OpenAIOrganization("some-org"),
        )
        key_and_base_url = OpenAIResponses(
            uid="test-model",
            model="some-model",
            api_key=OpenAIApiKey("some-value"),
            base_url="some-url",
        )

        with_client = OpenAIResponses(
            uid="test-model",
            model="some-model",
            client=openai_client,  # type: ignore
        )

        # these should all fail

        # no key and no client
        with pytest.raises(AssertionError):
            _ = OpenAIResponses(uid="test-model", model="some-model")

        # base_url and organization
        with pytest.raises(AssertionError):
            _ = OpenAIResponses(
                uid="test-model",
                model="some-model",
                organization=OpenAIOrganization("some-org"),
                base_url="some-url",
            )

    def test_translate_request(self, client):
        prompt = TextPrompt(text="some-text")
        request = client.translate_text_prompt(prompt, ModelOptions(max_tokens=100))
        assert request == OpenAIResponsesRequest(
            model="some-model",
            input=[OpenAIChatMessage(content="some-text", role="user")],
            max_output_tokens=100,
        )

    def test_translate_request_logprobs(self, client):
        prompt = TextPrompt(text="some-text")
        request = client.translate_text_prompt(prompt, ModelOptions(top_logprobs=2))
        assert request == OpenAIResponsesRequest(
            model="some-model",
            input=[OpenAIChatMessage(content="some-text", role="user")],
            max_output_tokens=None,
            top_logprobs=2,
            include=["message.output_text.logprobs"],
        )

    def test_translate_request_excessive_logprobs(self, client):
        # Set value above limit of 20
        prompt = TextPrompt(text="some-text")
        request = client.translate_text_prompt(prompt, ModelOptions(top_logprobs=21))
        assert request == OpenAIResponsesRequest(
            model="some-model",
            input=[OpenAIChatMessage(content="some-text", role="user")],
            max_output_tokens=None,
            top_logprobs=20,
            include=["message.output_text.logprobs"],
        )

    def test_translate_response(self, client):
        request = OpenAIResponsesRequest(model="some-model", input=[])
        response = _make_response("Hello there, how may I assist you today?")
        result = client.translate_response(request, response)
        assert result == SUTResponse(text="Hello there, how may I assist you today?", top_logprobs=None)

    def test_translate_response_logprobs(self, client):
        request = OpenAIResponsesRequest(model="some-model", input=[], top_logprobs=2)
        response = _make_response(
            "Hello!",
            logprobs=[
                {
                    "token": "Hello",
                    "logprob": -0.10257129,
                    "bytes": [72, 101, 108, 108, 111],
                    "top_logprobs": [
                        {"token": "Hello", "logprob": -0.10257129, "bytes": [72, 101, 108, 108, 111]},
                        {"token": "Hi", "logprob": -2.349693, "bytes": [72, 105]},
                    ],
                },
                {
                    "token": "!",
                    "logprob": -0.009831643,
                    "bytes": [33],
                    "top_logprobs": [
                        {"token": "!", "logprob": -0.009831643, "bytes": [33]},
                        {"token": " there", "logprob": -4.699771, "bytes": [32, 116, 104, 101, 114, 101]},
                    ],
                },
            ],
        )
        result = client.translate_response(request, response)
        assert result == SUTResponse(
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

    def test_translate_request_includes_temperature_when_accepted(self, client):
        """When _accepts_temperature is True, temperature from options is included in the request."""
        assert client._accepts_temperature is True
        prompt = TextPrompt(text="some-text")
        request = client.translate_text_prompt(prompt, ModelOptions(temperature=0.7))
        assert request.temperature == 0.7

    def test_translate_request_omits_temperature_when_not_accepted(self, client):
        """When _accepts_temperature is False, temperature is excluded from the request even if provided."""
        client._accepts_temperature = False
        prompt = TextPrompt(text="some-text")
        request = client.translate_text_prompt(prompt, ModelOptions(temperature=0.7))
        assert request.temperature is None
