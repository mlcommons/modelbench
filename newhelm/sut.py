from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from newhelm.placeholders import Prompt


@dataclass(frozen=True)
class SUTResponse:
    """The data that came out of the SUT."""

    completion: str


class SUT(ABC):
    """Base class for all SUTs. There is no guaranteed interface between SUTs, so no methods here."""

    pass


class PromptResponseSUT(SUT, ABC):
    """The base class for any SUT that is designed for handling a single-turn."""

    @abstractmethod
    def evaluate(self, prompt: Prompt) -> SUTResponse:
        pass
