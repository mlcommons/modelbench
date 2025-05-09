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
from modelgauge.pipeline import CachingPipe, Pipe, Sink, Source
from modelgauge.prompt import TextPrompt
from modelgauge.prompt_pipeline import PromptOutput, SutInteraction
from modelgauge.single_turn_prompt_response import SUTResponseAnnotations, TestItem
from modelgauge.sut import PromptResponseSUT, SUTResponse

logger = logging.getLogger(__name__)

ANNOTATOR_CSV_INPUT_COLUMNS = ["UID", "Prompt", "SUT", "Response"]


class AnnotatorInput(metaclass=ABCMeta):
    @abstractmethod
    def __iter__(self) -> Iterable[SutInteraction]:
        pass

    def __len__(self):
        count = 0
        for prompt in self:
            count += 1
        return count


class CsvAnnotatorInput(AnnotatorInput):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self._validate_file()

    def __iter__(self) -> Iterable[SutInteraction]:
        with open(self.path, newline="") as f:
            csvreader = csv.DictReader(f)
            for row in csvreader:
                prompt = TestItem(
                    prompt=TextPrompt(text=row["Prompt"]),
                    # Forward the underlying id to help make data tracking easier.
                    source_id=row["UID"],
                    # Context can be any type you want.
                    context=row,
                )
                response = SUTResponse(text=row["Response"])
                yield SutInteraction(prompt, row["SUT"], response)

    def _validate_file(self):
        with open(self.path, newline="") as f:
            csvreader = csv.reader(f)
            columns = next(csvreader)
        assert all(
            c in columns for c in ANNOTATOR_CSV_INPUT_COLUMNS
        ), f"Invalid input file. Must have columns: {', '.join(ANNOTATOR_CSV_INPUT_COLUMNS)}."


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

    def write(self, item: SutInteraction, results):
        if not isinstance(item.prompt.prompt, TextPrompt):
            raise Exception(f"Error handling {item}. Can only handle TextPrompts.")
        output_obj = {
            "UID": item.prompt.source_id,
            "Prompt": item.prompt.prompt.text,
            "SUT": item.sut_uid,
            "Response": item.response.text,
            "Annotations": results,
        }
        self.writer.write(output_obj)


class AnnotatorSource(Source):
    def __init__(self, input: AnnotatorInput):
        super().__init__()
        self.input = input

    def new_item_iterable(self):
        return self.input


class AnnotatorAssigner(Pipe):
    def __init__(self, annotators: dict[str, Annotator]):
        super().__init__()
        self.annotators = annotators

    def handle_item(self, item: SutInteraction):
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
        return sut_interaction, annotator_uid, result


class EnsembleVoter(Pipe):
    def __init__(self, ensemble: AnnotatorSet):
        super().__init__()
        self.ensemble = ensemble
        self.annotations: dict = defaultdict(dict)  # sut_interaction -> annotator uid -> annotations
        self.num_ensemble_votes = 0

    def handle_item(self, item):
        # Always pass the original item through
        self.downstream_put(item)
        sut_interaction, annotator_uid, annotation = item
        if annotator_uid in self.ensemble.annotators:
            self.annotations[sut_interaction][annotator_uid] = annotation
            if len(self.annotations[sut_interaction]) == len(self.ensemble.annotators):
                # All annotators have responded, so we can compute the ensemble response.
                annotations = {k: Annotation.from_instance(v) for k, v in self.annotations[sut_interaction].items()}
                result = self.ensemble.evaluate(
                    SUTResponseAnnotations(
                        test_item=sut_interaction.prompt, sut_response=sut_interaction.response, annotations=annotations
                    )
                )
                self.downstream_put((sut_interaction, "ensemble", result))
                self.num_ensemble_votes += 1


class AnnotatorSink(Sink):
    unfinished: defaultdict[SutInteraction, dict[str, str]]

    def __init__(self, annotators: dict[str, Annotator], writer: JsonlAnnotatorOutput, ensemble: bool = False):
        super().__init__()
        self.annotators = annotators
        self.ensemble = ensemble
        self.writer = writer
        self.unfinished = defaultdict(lambda: dict())

    def run(self):
        with self.writer:
            super().run()

    def interaction_is_complete(self, sut_interaction) -> bool:
        num_expected_annotations = len(self.annotators)
        if self.ensemble:
            num_expected_annotations += 1
        return len(self.unfinished[sut_interaction]) == num_expected_annotations

    def handle_item(self, item):
        sut_interaction, annotator_uid, annotation = item
        if isinstance(annotation, BaseModel):
            annotation = annotation.model_dump()
        self.unfinished[sut_interaction][annotator_uid] = annotation
        if self.interaction_is_complete(sut_interaction):
            self.writer.write(sut_interaction, self.unfinished[sut_interaction])
            self._debug(f"wrote {sut_interaction}")
            del self.unfinished[sut_interaction]
