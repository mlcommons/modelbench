from enum import Enum
from pydantic import BaseModel
from typing import List


class ChatRole(str, Enum):
    user = "USER"
    sut = "SUT"
    system = "SYSTEM"


class ChatMessage(BaseModel):
    text: str
    role: ChatRole


class ChatPrompt(BaseModel):
    messages: List[ChatMessage]


class TextPrompt(BaseModel, frozen=True):
    """What actually goes to the SUT."""

    text: str
