from modelgauge.prompt import ChatMessage, ChatPrompt, ChatRole
from modelgauge.prompt_formatting import format_chat


def test_format_chat_just_user():
    chat = ChatPrompt(messages=[ChatMessage(text="some-text", role=ChatRole.user)])
    assert (
        format_chat(chat)
        == """\
user: some-text

assistant: """
    )


def test_format_chat_multi_turn():
    chat = ChatPrompt(
        messages=[
            ChatMessage(text="first-text", role=ChatRole.sut),
            ChatMessage(text="second-text", role=ChatRole.user),
        ]
    )
    assert (
        format_chat(chat)
        == """\
assistant: first-text

user: second-text

assistant: """
    )


def test_format_chat_override_names():
    chat = ChatPrompt(
        messages=[
            ChatMessage(text="first-text", role=ChatRole.sut),
            ChatMessage(text="second-text", role=ChatRole.user),
        ]
    )
    assert (
        format_chat(chat, user_role="human", sut_role="bot")
        == """\
bot: first-text

human: second-text

bot: """
    )
