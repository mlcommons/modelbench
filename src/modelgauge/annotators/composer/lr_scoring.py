import hashlib
from typing import Sequence

import numpy as np

from modelgauge.annotators.composer.context import EvalContext
from modelgauge.annotators.composer.nodes import (
    CacheableNodeMixin,
    NodeOutput,
)

from airrlogger.log_config import get_logger

from evaluators.components.hazard_response_classifier import (
    HazardResponseClassifier,
    ENABLEMENT_NAME,
    LEGITIMIZATION_NAME,
)
from evaluators.components.sentence_enricher import Sentence, SentenceEnricher


logger = get_logger(__name__)


class LogisticRegressionScoringNode(SentenceEnricher, CacheableNodeMixin):
    def __init__(
        self,
        name: str,
        routes: Sequence[str],
        embedding_node_name: str,
        classifier: HazardResponseClassifier,
    ) -> None:
        if (ENABLEMENT_NAME not in name and LEGITIMIZATION_NAME not in name) or (
            ENABLEMENT_NAME in name and LEGITIMIZATION_NAME in name
        ):
            raise ValueError(
                f"Invalid component name: {name}. Must contain {ENABLEMENT_NAME} xor {LEGITIMIZATION_NAME}."
            )
        self.embedding_node_name = embedding_node_name
        self.classifier = classifier
        super().__init__(name, routes=routes)

    @property
    def component_name(self) -> str:
        return ENABLEMENT_NAME if ENABLEMENT_NAME in self.name else LEGITIMIZATION_NAME

    def cache_key(self, ctx: EvalContext) -> str:
        sentences = self._get_sentences(ctx)
        vectors = np.ascontiguousarray(self._get_embedding_vectors(ctx))
        hazard = self._get_hazard(ctx)
        sentence_payload = "\x00".join(
            "\x01".join(
                (
                    sentence.text,
                    str(sentence.is_prompt_repetition),
                    str(sentence.later_response_authored_continuation),
                )
            )
            for sentence in sentences
        )
        payload = (
            f"{self.component_name}\x00{hazard}\x00{vectors.dtype}\x00{vectors.shape}\x00"
            f"{sentence_payload}".encode("utf-8") + b"\x00" + vectors.tobytes()
        )
        return hashlib.blake2b(payload, digest_size=16).hexdigest()

    def _get_embedding_vectors(self, ctx: EvalContext) -> np.ndarray:
        vectors = ctx.ancestor_output(self.embedding_node_name)
        assert (
            vectors is not None
        ), f"No parent output found for embedding node name {self.embedding_node_name}."
        return vectors.value

    def _get_hazard(self, ctx: EvalContext) -> str:
        # TODO: Replace with output of hazard detector node if that gets implemented.
        hazard = ctx.metadata.get("hazard")
        assert hazard is not None, f"Hazard not found in metadata."
        return hazard

    def _filter_prompt_repetitions_from_vectors(
        self, sentences: list[Sentence], vectors: np.ndarray
    ) -> np.ndarray:
        selected_indices = tuple(
            index
            for index, sentence in enumerate(sentences)
            if not sentence.is_prompt_repetition
            or sentence.later_response_authored_continuation
        )
        return vectors[np.asarray(selected_indices, dtype=int)]

    def run(self, ctx: EvalContext) -> NodeOutput:
        sentences = self._get_sentences(ctx)
        vectors = self._get_embedding_vectors(ctx)
        if vectors.shape[0] != len(sentences):
            raise ValueError(
                "sentence metadata and embedding rows disagree: "
                f"{len(sentences)} != {vectors.shape[0]}"
            )
        if self.component_name == ENABLEMENT_NAME:
            vectors = self._filter_prompt_repetitions_from_vectors(sentences, vectors)

        hazard = self._get_hazard(ctx)
        evaluated_hazards = (hazard,)
        per_hazard = {}
        for hazard in evaluated_hazards:
            judgment = self.classifier.judge_hazard_for_component(
                self.component_name, hazard, vectors
            )
            per_hazard[hazard] = judgment

        return NodeOutput(value=per_hazard, original_ctx=ctx, updated_ctx=ctx)
