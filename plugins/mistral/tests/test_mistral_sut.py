import pytest
from mistralai.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ClassificationObject,
    ClassificationResponse,
    UsageInfo,
)
from modelgauge.prompt import SUTOptions, TextPrompt
from modelgauge.sut import SUTCompletion, SUTResponse
from modelgauge.suts.mistral_client import MistralAIAPIKey
from modelgauge.suts.mistral_sut import (
    MistralAIModeratedSut,
    MistralAIResponse,
    MistralAIResponseWithModerations,
    MistralAISut,
)


@pytest.fixture
def req():
    return {
        "model": "ministral-8b-latest",
        "messages": [{"role": "user", "content": "Why did the chicken cross the road?"}],
        "max_tokens": 91,
        "temperature": 0.3,
        "n": 1,
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
    return MistralAISut("ministral-8b-latest", "ministral-8b-latest", MistralAIAPIKey("fake"))


@pytest.fixture
def moderated_req():
    return {
        "model": "ministral-8b-latest",
        "messages": [{"role": "user", "content": "Why did the chicken cross the road?"}],
        "max_tokens": 91,
        "temperature": 0.3,
        "n": 2,
    }


def _make_moderation(score):
    return ClassificationResponse(
        id="fake-id",
        model="moderator",
        results=[
            ClassificationObject(
                categories={"hazard_1": False, "hazard_2": False}, category_scores={"hazard_1": 0.1, "hazard_2": 0.2}
            )
        ],
    )


@pytest.fixture
def moderated_response():
    response = MistralAIResponse(
        id="ed6c8eccd53e4b319a7bc566f6a53357",
        object="chat.completion",
        model="ministral-8b-latest",
        created=1731977771,
        usage=UsageInfo(prompt_tokens=11, completion_tokens=22, total_tokens=33),
        choices=[
            ChatCompletionChoice(
                index=0,
                message=AssistantMessage(
                    content="Response 1",
                    tool_calls=None,
                    prefix=False,
                    role="assistant",
                ),
                finish_reason="stop",
            ),
            ChatCompletionChoice(
                index=0,
                message=AssistantMessage(
                    content="Response 2",
                    tool_calls=None,
                    prefix=False,
                    role="assistant",
                ),
                finish_reason="stop",
            ),
        ],
    )
    return MistralAIResponseWithModerations(
        response=response, moderations={0: _make_moderation(0.1), 1: _make_moderation(0.2)}
    )


@pytest.fixture
def moderated_sut():
    return MistralAIModeratedSut(
        "ministral-8b-latest", "ministral-8b-latest", "moderator", 2, 0.3, 0.3, MistralAIAPIKey("fake")
    )


class TestMistralAISut:

    def test_request(self, sut, req):
        translated_req = sut.translate_text_prompt(
            TextPrompt(text="Why did the chicken cross the road?", options=SUTOptions(temperature=0.3, max_tokens=91))
        )
        assert translated_req.model_dump(exclude_none=True) == req

    def test_response(self, sut, req, response):
        resp = sut.translate_response(request=req, response=response)
        assert resp == SUTResponse(completions=[SUTCompletion(text="The classic joke has several variations")])


class TestMistralAIModeratedSut:

    @pytest.mark.parametrize("prompt_temp,prompt_num_completions", [(None, None), (0.3, 3), (0.1, 1000)])
    def test_request(self, moderated_sut, moderated_req, prompt_temp, prompt_num_completions):
        translated_req = moderated_sut.translate_text_prompt(
            TextPrompt(
                text="Why did the chicken cross the road?",
                options=SUTOptions(temperature=prompt_temp, max_tokens=91),
                num_completions=prompt_num_completions,
            )
        )
        assert translated_req.model_dump(exclude_none=True) == moderated_req

    def test_response(self, moderated_sut, moderated_req, moderated_response):
        resp = moderated_sut.translate_response(request=moderated_req, response=moderated_response)
        assert resp == SUTResponse(completions=[SUTCompletion(text="Response 1")])

    def test_response_over_safety_threshold(self, moderated_req, moderated_response):
        sut = MistralAIModeratedSut(
            "ministral-8b-latest", "ministral-8b-latest", "moderator", 2, 0.3, 0.001, MistralAIAPIKey("fake")
        )
        resp = sut.translate_response(request=moderated_req, response=moderated_response)
        assert resp == SUTResponse(completions=[SUTCompletion(text="I'm sorry I cannot assist with this request.")])
