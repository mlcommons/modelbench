from enum import Enum
from typing import List

from pydantic import BaseModel


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
