from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

from pydantic import BaseModel

from newhelm.annotation import Annotation
from newhelm.single_turn_prompt_response import PromptInteraction, PromptWithContext
from newhelm.sut import SUTCompletion
from newhelm.typed_data import TypedData


AnnotationType = TypeVar("AnnotationType", bound=BaseModel)


class BaseAnnotator(ABC):
    """The base class for all annotators."""

    pass


class CompletionAnnotator(BaseAnnotator, Generic[AnnotationType]):
    @abstractmethod
    def translate_request(self, prompt: PromptWithContext, completion: SUTCompletion):
        pass

    @abstractmethod
    def annotate(self, annotation_request):
        """Returns an annotation for a single TestItem's interactions."""
        pass

    @abstractmethod
    def translate_response(self, request, response) -> AnnotationType:
        pass
