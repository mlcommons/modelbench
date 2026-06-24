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


class TextPromptWithMetadata(TextPrompt, frozen=True):
    """TextPrompt with additional metadata. The
    metadata is NOT sent to the SUT, but is passed to the evaluator
    for benchmarking."""

    metadata: dict
