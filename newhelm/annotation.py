from abc import ABC
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class Annotation(ABC):
    """Generic class extended by all annotators.

    Every annotator can return data however it wants.
    Since Tests are responsible for both deciding what
    Annotators to apply and how to interpret their results,
    they can cast to the proper type.
    """

    pass
