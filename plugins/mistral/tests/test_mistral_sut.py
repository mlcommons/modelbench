import pytest
from mistralai.models import AssistantMessage, ChatCompletionChoice, UsageInfo
from modelgauge.prompt import SUTOptions, TextPrompt
from modelgauge.sut import SUTCompletion, SUTResponse
from modelgauge.suts.mistral_client import MistralAIAPIKey
from modelgauge.suts.mistral_sut import MistralAIResponse, MistralAISut


@pytest.fixture
def req():
    return {
        "model": "ministral-8b-latest",
        "messages": [{"role": "user", "content": "Why did the chicken cross the road?"}],
        "max_tokens": 91,
        "temperature": 0.3,
    }


@pytest.fixture
def response():
    return MistralAIResponse(
        id="ed6c8eccd53e4b319a7bc566f6a53357",
        object="chat.completion",
        model="ministral-8b-latest",
        created=1731977771,
        usage=UsageInfo(prompt_tokens=11, completion_tokens=22, total_tokens=33),
        choices=[
            ChatCompletionChoice(
                index=0,
                message=AssistantMessage(
                    content="The classic joke has several variations",
                    tool_calls=None,
                    prefix=False,
                    role="assistant",
                ),
                finish_reason="stop",
            )
        ],
    )


@pytest.fixture
def sut():
    return MistralAISut("ministral-8b-latest", "ministral-8b-latest", "latest", MistralAIAPIKey("fake"))


class TestMistralAISut:

    def test_request(self, sut, req):
        translated_req = sut.translate_text_prompt(
            TextPrompt(text="Why did the chicken cross the road?", options=SUTOptions(temperature=0.3, max_tokens=91))
        )
        assert translated_req.model_dump(exclude_none=True) == req

    def test_response(self, sut, req, response):
        resp = sut.translate_response(request=req, response=response)
        assert resp == SUTResponse(completions=[SUTCompletion(text="The classic joke has several variations")])
