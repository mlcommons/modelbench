from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Callable, List


@dataclass(frozen=True)
class Prompt:
    """What actually goes to the SUT."""

    text: str


@dataclass(frozen=True)
class Measurement:
    """A numeric representation of the quality of a single TestItem.

    Aggregating together all Measurement's with the same name should produce the Test's Result.
    """

    name: str
    value: float


@dataclass(frozen=True)
class Result:
    """The measurement produced by Test."""

    # Just a placeholder.
    name: str
    value: float
