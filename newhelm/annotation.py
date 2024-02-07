from abc import ABC
from typing import List

from pydantic import BaseModel


class Annotation(ABC, BaseModel):
    """Generic class extended by all annotators.

    Every annotator can return data however it wants.
    Since Tests are responsible for both deciding what
    Annotators to apply and how to interpret their results,
    they can cast to the proper type.
    """

    pass
