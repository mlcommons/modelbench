from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, TypeVar

from newhelm.placeholders import Prompt

RequestType = TypeVar("RequestType")
ResponseType = TypeVar("ResponseType")


@dataclass(frozen=True)
class SUTResponse:
    """The data that came out of the SUT."""

    completion: str


class SUT(ABC):
    """Base class for all SUTs. There is no guaranteed interface between SUTs, so no methods here."""

    pass


class PromptResponseSUT(SUT, ABC, Generic[RequestType, ResponseType]):
    """The base class for any SUT that is designed for handling a single-turn."""

    @abstractmethod
    def translate_request(self, prompt: Prompt) -> RequestType:
        pass

    @abstractmethod
    def evaluate(self, request: RequestType) -> ResponseType:
        pass

    @abstractmethod
    def translate_response(self, prompt: Prompt, response: ResponseType) -> SUTResponse:
        pass
