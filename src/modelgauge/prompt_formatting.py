from modelgauge.prompt import ChatPrompt, ChatRole


def format_chat(
    chat: ChatPrompt, *, user_role: str = "user", sut_role: str = "assistant"
) -> str:
    """Flattens a chat conversation into a single text prompt"""
    blocks = []
    for message in chat.messages:
        role_text: str
        if message.role == ChatRole.user:
            role_text = user_role
        else:
            role_text = sut_role
        blocks.append(f"{role_text}: {message.text}")
    blocks.append(f"{sut_role}: ")
    return "\n\n".join(blocks)
