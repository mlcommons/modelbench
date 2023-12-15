from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from newhelm.placeholders import Prompt, PromptTemplate


@dataclass(frozen=True)
class Turn:
    """This is mostly a placeholder"""

    prompt: Prompt
    """The prompt sent to the SUT."""

    completion: str
    """The data that came back from the SUT."""


@dataclass(frozen=True)
class Interaction:
    turns: List[Turn]


class PromptResponseSUT(ABC):
    """The base class for any SUT that is designed for handling a single-turn."""

    @abstractmethod
    def specialize(self, prompt_template: PromptTemplate) -> Prompt:
        pass

    @abstractmethod
    def evaluate(self, prompt: Prompt) -> Interaction:
        pass
