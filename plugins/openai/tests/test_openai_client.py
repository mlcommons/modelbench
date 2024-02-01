from newhelm.suts.openai_client import (
    OpenAIChat,
    OpenAIChatMessage,
    OpenAIChatRequest,
)

from newhelm.placeholders import Prompt


def test_openai_chat_translate_request():
    client = OpenAIChat("some-model")
    prompt = Prompt("some-text")
    request = client.translate_request(prompt)
    assert request == OpenAIChatRequest(
        model="some-model",
        messages=[OpenAIChatMessage(content="some-text", role="user")],
        frequency_penalty=0,
        max_tokens=100,
        n=1,
        presence_penalty=0,
        stop=[],
        temperature=1.0,
        top_p=1,
    )


# TODO more testing.
