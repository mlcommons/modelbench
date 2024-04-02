from abc import ABC
from newhelm.typed_data import TypedData
from typing import List


class Annotation(TypedData):
    """Container for plugin defined annotation data.

    Every annotator can return data however it wants.
    Since Tests are responsible for both deciding what
    Annotators to apply and how to interpret their results,
    they can use `to_instance` to get it back in the form they want.
    """

    pass
