from typing import List
from pydantic import BaseModel

from newhelm.base_annotator import BaseAnnotator
from newhelm.single_turn_prompt_response import PromptInteraction


class DemoYBadAnnotation(BaseModel):
    """How bad each SUTResponse in the TestItem is."""

    badness: List[int]


class DemoYBadAnnotator(BaseAnnotator[DemoYBadAnnotation]):
    """A demonstration annotator that dislikes the letter Y.

    Real Annotators are intended to do expensive processing on the string,
    such as calling another model collecting data from human raters. For
    the demo though, we want something cheap and deterministic.
    """

    def annotate_test_item(
        self, interactions: List[PromptInteraction]
    ) -> DemoYBadAnnotation:
        badness = []
        for interaction in interactions:
            worst = 0
            for completion in interaction.response.completions:
                score = 0
                for character in completion.text:
                    if character in {"Y", "y"}:
                        score += 1
                worst = max(worst, score)
            badness.append(worst)
        return DemoYBadAnnotation(badness=badness)
