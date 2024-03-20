from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

from pydantic import BaseModel

from newhelm.prompt import ChatPrompt, TextPrompt
from newhelm.record_init import InitializationRecord
from newhelm.tracked_object import TrackedObject

RequestType = TypeVar("RequestType")
ResponseType = TypeVar("ResponseType")


class SUTCompletion(BaseModel):
    """All data about a single completion in the response."""

    text: str


class SUTResponse(BaseModel):
    """The data that came out of the SUT."""

    completions: List[SUTCompletion]


class SUT(TrackedObject):
    """Base class for all SUTs. There is no guaranteed interface between SUTs, so no methods here."""

    def __init__(self, uid: str):
        super().__init__(uid)
        # The initialization record is set automatically by @newhelm_test()
        self.initialization_record: InitializationRecord


class PromptResponseSUT(SUT, ABC, Generic[RequestType, ResponseType]):
    """The base class for any SUT that is designed for handling a single-turn."""

    @abstractmethod
    def translate_text_prompt(self, prompt: TextPrompt) -> RequestType:
        pass

    @abstractmethod
    def translate_chat_prompt(self, prompt: ChatPrompt) -> RequestType:
        pass

    @abstractmethod
    def evaluate(self, request: RequestType) -> ResponseType:
        pass

    @abstractmethod
    def translate_response(
        self, request: RequestType, response: ResponseType
    ) -> SUTResponse:
        pass
