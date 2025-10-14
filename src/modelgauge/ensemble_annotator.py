from typing import Any

from modelgauge.annotation import EnsembleSafetyAnnotation
from modelgauge.annotator import Annotator
from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.config import load_secrets_from_config, raise_if_missing_from_config
from modelgauge.ensemble_strategies import ENSEMBLE_STRATEGIES
from modelgauge.prompt import ChatPrompt, TextPrompt
from modelgauge.secret_values import MissingSecretValues
from modelgauge.sut import SUTResponse


class EnsembleAnnotator(Annotator):
    """Defines an ensemble; responds like an annotator."""

    def __init__(self, uid, annotators: list[str], ensemble_strategy: str):
        super().__init__(uid)
        self.annotators = self._make_annotators(annotators)
        if ensemble_strategy not in ENSEMBLE_STRATEGIES:
            raise ValueError(f"Ensemble strategy {ensemble_strategy} not recognized.")
        self.ensemble_strategy = ENSEMBLE_STRATEGIES[ensemble_strategy]

    def _make_annotators(self, annotator_uids: list[str]) -> dict[str, Annotator]:
        secrets = load_secrets_from_config()
        missing_secrets: list[MissingSecretValues] = []
        for annotator_uid in annotator_uids:
            missing_secrets.extend(ANNOTATORS.get_missing_dependencies(annotator_uid, secrets=secrets))
        raise_if_missing_from_config(missing_secrets)

        annotators = {uid: ANNOTATORS.make_instance(uid, secrets=secrets) for uid in annotator_uids}
        return annotators  # type: ignore

    def translate_prompt(self, prompt: TextPrompt | ChatPrompt, response: SUTResponse):
        return {uid: annotator.translate_prompt(prompt, response) for uid, annotator in self.annotators.items()}

    def annotate(self, annotation_request: dict[str, Any]):
        return {uid: annotator.annotate(annotation_request[uid]) for uid, annotator in self.annotators.items()}

    def translate_response(self, request: dict[str, Any], response: dict[str, Any]):
        annotations = {
            uid: annotator.translate_response(request[uid], response[uid]) for uid, annotator in self.annotators.items()
        }
        ensemble_annotation = self.ensemble_strategy.compute_response(annotations)
        return EnsembleSafetyAnnotation(
            is_safe=ensemble_annotation.is_safe,
            is_valid=ensemble_annotation.is_valid,
            joined_responses=annotations,
            metadata=ensemble_annotation.metadata,  # TODO: Merge metadata here instead of in strategy
        )
