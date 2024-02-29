from typing import List
from pydantic import BaseModel
from newhelm.base_annotator import BaseAnnotator
from newhelm.single_turn_prompt_response import PromptInteraction


class FakeAnnotation(BaseModel):
    sut_text: str


class FakeAnnotator(BaseAnnotator[FakeAnnotation]):
    """Fake annotator that just returns the first completion from the SUT."""

    def __init__(self):
        self.annotate_test_item_calls = 0

    def annotate_test_item(
        self, interactions: List[PromptInteraction]
    ) -> FakeAnnotation:
        self.annotate_test_item_calls += 1
        """Returns an annotation for a single TestItem's interactions."""
        return FakeAnnotation(sut_text=interactions[0].response.completions[0].text)
