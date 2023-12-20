from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Callable, List


@dataclass(frozen=True)
class Prompt:
    """What actually goes to the SUT."""

    text: str


@dataclass(frozen=True)
class Result:
    """The measurement produced by Test."""

    # Just a placeholder.
    name: str
    value: float
