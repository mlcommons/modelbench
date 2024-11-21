import pytest
from modelgauge.prompt import SUTOptions, TextPrompt
from modelgauge.sut import SUTCompletion, SUTResponse
from modelgauge.suts.vertexai_client import VertexAIProjectId, VertexAIRegion
from modelgauge.suts.vertexai_mistral_sut import (
    VertexAIMistralAISut,
    VertexAIMistralResponse,
)


@pytest.fixture
def req():
    return {
        "model": "mistral-large",
        "stream": False,
        "messages": [{"role": "user", "content": "Why did the chicken cross the road?"}],
        "n": 1,
        "safe_prompt": True,
        "max_tokens": 17,
        "temperature": 0.5,
    }


@pytest.fixture
def response():
    return VertexAIMistralResponse(
        id="ed6c8eccd53e4b319a7bc566f6a53357",
        object="chat.completion",
        model="mistral-large",
        created=1731977771,
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "To get to the other side!",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        usage={"prompt_tokens": 11, "total_tokens": 62, "completion_tokens": 51},
    )


@pytest.fixture
def sut():
    return VertexAIMistralAISut(
        "vertexai-mistral-large-2407",
        "mistral-large",
        "2407",
        project_id=VertexAIProjectId("fake"),
        region=VertexAIRegion("us-central1"),
    )


class TestMistralAISut:

    def test_request(self, sut, req):
        translated_req = sut.translate_text_prompt(
            TextPrompt(text="Why did the chicken cross the road?", options=SUTOptions(temperature=0.5, max_tokens=17))
        )
        assert translated_req.model_dump(exclude_none=True) == req

    def test_response(self, sut, req, response):
        resp = sut.translate_response(request=req, response=response)
        assert resp == SUTResponse(completions=[SUTCompletion(text="To get to the other side!")])
