from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

from newhelm.single_turn_prompt_response import PromptInteraction


AnnotationType = TypeVar("AnnotationType")


class BaseAnnotator(ABC, Generic[AnnotationType]):
    """The base class for all annotators."""

    @abstractmethod
    def annotate_test_item(
        self, interactions: List[PromptInteraction]
    ) -> AnnotationType:
        """Returns an annotation for a single TestItem's interactions."""
        pass
