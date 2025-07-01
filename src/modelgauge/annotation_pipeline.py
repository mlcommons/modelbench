import csv
import jsonlines
import logging
import time
from abc import abstractmethod, ABCMeta
from collections import defaultdict
from pydantic import BaseModel
from typing import Iterable

from modelgauge.annotation import Annotation
from modelgauge.annotator import Annotator
from modelgauge.annotator_set import AnnotatorSet
from modelgauge.dataset import PromptResponseDataset
from modelgauge.data_schema import DEFAULT_PROMPT_RESPONSE_SCHEMA, PromptResponseSchema
from modelgauge.pipeline import CachingPipe, Pipe, Sink, Source
from modelgauge.prompt import TextPrompt
from modelgauge.prompt_pipeline import PromptOutput
from modelgauge.single_turn_prompt_response import (
    AnnotatedSUTInteraction,
    SUTResponseAnnotations,
    SUTInteraction,
    TestItem,
)
from modelgauge.sut import PromptResponseSUT, SUTResponse

logger = logging.getLogger(__name__)


class JsonlAnnotatorOutput(PromptOutput):
    def __init__(self, path):
        super().__init__()
        assert path.suffix.lower() == ".jsonl", f"Invalid output file {path}. Must be of type JSONL."

        self.path = path
        self.file = None
        self.writer = None

    def __enter__(self):
        self.file = open(self.path, "w", newline="")
        self.writer = jsonlines.Writer(self.file)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()
        self.file.close()

    def write(self, item: SUTInteraction, results):
        if not isinstance(item.prompt.prompt, TextPrompt):
            raise Exception(f"Error handling {item}. Can only handle TextPrompts.")
        # TODO: Standardize annotation schema.
        output_obj = {
            DEFAULT_PROMPT_RESPONSE_SCHEMA.prompt_uid: item.prompt.source_id,
            DEFAULT_PROMPT_RESPONSE_SCHEMA.prompt_text: item.prompt.prompt.text,
            DEFAULT_PROMPT_RESPONSE_SCHEMA.sut_uid: item.sut_uid,
            DEFAULT_PROMPT_RESPONSE_SCHEMA.sut_response: item.response.text,
            "Annotations": results,
        }
        self.writer.write(output_obj)


class AnnotatorSource(Source):
    def __init__(self, input: PromptResponseDataset):
        super().__init__()
        self.input = input

    def new_item_iterable(self):
        return self.input


class AnnotatorAssigner(Pipe):
    def __init__(self, annotators: dict[str, Annotator]):
        super().__init__()
        self.annotators = annotators

    def handle_item(self, item: SUTInteraction):
        for annotator_uid in self.annotators:
            self.downstream_put((item, annotator_uid))


class AnnotatorWorkers(CachingPipe):
    def __init__(self, annotators: dict[str, Annotator], workers=None, cache_path=None):
        self.sleep_time = 10
        if workers is None:
            workers = 8
        super().__init__(thread_count=workers, cache_path=cache_path)
        self.annotators = annotators
        self.annotation_counts = {uid: 0 for uid in annotators}

    def key(self, item):
        sut_interaction, annotator_uid = item
        annotator = self.annotators[annotator_uid]
        request = annotator.translate_request(sut_interaction.prompt, sut_interaction.response)
        if isinstance(request, BaseModel):
            request = request.model_dump_json()
        return (request, annotator_uid)

    def handle_uncached_item(self, item):
        sut_interaction, annotator_uid = item
        annotator = self.annotators[annotator_uid]
        request = annotator.translate_request(sut_interaction.prompt, sut_interaction.response)
        tries = 0
        while True:
            tries += 1
            try:
                response = annotator.annotate(request)
                break
            except Exception as e:
                logger.warning(
                    f"Exception calling annotator {annotator_uid} on attempt {tries}: {e}\nRetrying.....", exc_info=True
                )
                time.sleep(self.sleep_time)
        result = annotator.translate_response(request, response)
        self.annotation_counts[annotator_uid] += 1
        return AnnotatedSUTInteraction(annotator_uid=annotator_uid, annotation=result, sut_interaction=sut_interaction)


class EnsembleVoter(Pipe):
    def __init__(self, ensemble: AnnotatorSet):
        super().__init__()
        self.ensemble = ensemble
        self.annotations: dict = defaultdict(dict)  # sut_interaction -> annotator uid -> annotations
        self.num_ensemble_votes = 0

    def handle_item(self, item):
        # Always pass the original item through
        self.downstream_put(item)
        if item.annotator_uid in self.ensemble.annotators:
            self.annotations[item.sut_interaction][item.annotator_uid] = item.annotation
            if len(self.annotations[item.sut_interaction]) == len(self.ensemble.annotators):
                # All annotators have responded, so we can compute the ensemble response.
                annotations = {
                    k: Annotation.from_instance(v) for k, v in self.annotations[item.sut_interaction].items()
                }
                result = self.ensemble.evaluate(
                    SUTResponseAnnotations(
                        test_item=item.sut_interaction.prompt,
                        sut_response=item.sut_interaction.response,
                        annotations=annotations,
                    )
                )
                self.downstream_put(
                    AnnotatedSUTInteraction(
                        annotator_uid="ensemble", annotation=result, sut_interaction=item.sut_interaction
                    )
                )
                self.num_ensemble_votes += 1


class AnnotatorSink(Sink):
    unfinished: defaultdict[SUTInteraction, dict[str, str]]

    def __init__(self, annotators: dict[str, Annotator], writer: JsonlAnnotatorOutput, ensemble: bool = False):
        super().__init__()
        self.annotators = annotators
        self.ensemble = ensemble
        self.writer = writer
        self.unfinished = defaultdict(lambda: dict())

    def run(self):
        with self.writer:
            super().run()

    def interaction_is_complete(self, sut_interaction: SUTInteraction) -> bool:
        num_expected_annotations = len(self.annotators)
        if self.ensemble:
            num_expected_annotations += 1
        return len(self.unfinished[sut_interaction]) == num_expected_annotations

    def handle_item(self, item: AnnotatedSUTInteraction):
        # Convert Pydantic model to dict if needed
        annotation = item.annotation.model_dump() if isinstance(item.annotation, BaseModel) else item.annotation
        self.unfinished[item.sut_interaction][item.annotator_uid] = annotation
        if self.interaction_is_complete(item.sut_interaction):
            self.writer.write(item.sut_interaction, self.unfinished[item.sut_interaction])
            self._debug(f"wrote {item.sut_interaction}")
            del self.unfinished[item.sut_interaction]
