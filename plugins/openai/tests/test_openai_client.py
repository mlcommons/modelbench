from newhelm.suts.openai_client import (
    OpenAIChat,
    OpenAIChatMessage,
    OpenAIChatRequest,
    OpenAIApiKey,
    OpenAIOrgId,
)

from newhelm.prompt import TextPrompt


def test_openai_chat_translate_request():
    client = OpenAIChat(
        model="some-model", api_key=OpenAIApiKey("some-value"), org_id=OpenAIOrgId(None)
    )
    prompt = TextPrompt(text="some-text")
    request = client.translate_text_prompt(prompt)
    assert request == OpenAIChatRequest(
        model="some-model",
        messages=[OpenAIChatMessage(content="some-text", role="user")],
        max_tokens=100,
        n=1,
    )


# TODO more testing.
