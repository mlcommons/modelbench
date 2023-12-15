from dataclasses import dataclass, field
from typing import List

from newhelm.sut import Interaction


@dataclass(frozen=True)
class Annotation:
    """Placeholder for data collected about the quality of an Interaction."""

    placeholder: str


@dataclass(frozen=True)
class AnnotatedInteraction:
    """An Interaction and all of the Annotations we've collected for it."""

    interaction: Interaction
    annotations: List[Annotation] = field(default_factory=list)
