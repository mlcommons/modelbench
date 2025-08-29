from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ReadyResponse:
    is_ready: bool
    response: Optional[Any] = None
    error: Optional[Exception] = None


@dataclass
class ReadyResponses:
    all_ready: bool
    responses: dict[str, ReadyResponse]

    @classmethod
    def from_dict(cls, responses: dict[str, ReadyResponse]) -> "ReadyResponses":
        all_ready = all(r.is_ready for r in responses.values())
        return cls(responses=responses, all_ready=all_ready)


class Readyable(ABC):
    def is_ready(self) -> ReadyResponse:
        try:
            return self.run_readiness_check()
        except Exception as e:
            return ReadyResponse(is_ready=False, error=e)

    @abstractmethod
    def run_readiness_check(self) -> ReadyResponse:
        pass
