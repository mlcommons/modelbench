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
    )


# TODO more testing.
