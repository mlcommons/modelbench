import csv
import jsonlines
import sys
import traceback
from abc import abstractmethod, ABCMeta
from collections import defaultdict
from pydantic import BaseModel
from typing import Iterable

from modelgauge.annotator import Annotator
from modelgauge.pipeline import CachingPipe, Pipe, Sink, Source
from modelgauge.prompt import TextPrompt
from modelgauge.prompt_pipeline import PromptOutput, SutInteraction
from modelgauge.single_turn_prompt_response import PromptWithContext
from modelgauge.sut import PromptResponseSUT, SUTCompletion

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
                prompt = PromptWithContext(
                    prompt=TextPrompt(text=row["Prompt"]),
                    # Forward the underlying id to help make data tracking easier.
                    source_id=row["UID"],
                    # Context can be any type you want.
                    context=row,
                )
                response = SUTCompletion(text=row["Response"])
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
        assert (
            path.suffix.lower() == ".jsonl"
        ), f"Invalid output file {path}. Must be of type JSONL."

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
        if workers is None:
            workers = 8
        super().__init__(thread_count=workers, cache_path=cache_path)
        self.annotators = annotators

    def key(self, item):
        sut_interaction, annotator_uid = item
        annotator = self.annotators[annotator_uid]
        request = annotator.translate_request(
            sut_interaction.prompt, sut_interaction.response
        )
        if isinstance(request, BaseModel):
            request = request.model_dump_json()
        return (request, annotator_uid)

    def handle_uncached_item(self, item):
        sut_interaction, annotator_uid = item
        try:
            annotator = self.annotators[annotator_uid]
            request = annotator.translate_request(
                sut_interaction.prompt, sut_interaction.response
            )
            response = annotator.annotate(request)
            result = annotator.translate_response(request, response)
            return sut_interaction, annotator_uid, result
        except Exception as e:
            print(
                f"unexpected failure processing {item} for {annotator_uid}.\n{e}\n",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)


class AnnotatorSink(Sink):
    unfinished: defaultdict[SutInteraction, dict[str, str]]

    def __init__(self, annotators: dict[str, Annotator], writer: JsonlAnnotatorOutput):
        super().__init__()
        self.annotators = annotators
        self.writer = writer
        self.unfinished = defaultdict(lambda: dict())

    def run(self):
        with self.writer:
            super().run()

    def handle_item(self, item):
        sut_interaction, annotator_uid, annotation = item
        if isinstance(annotation, BaseModel):
            annotation = annotation.model_dump()
        self.unfinished[sut_interaction][annotator_uid] = annotation
        if len(self.unfinished[sut_interaction]) == len(self.annotators):
            self.writer.write(sut_interaction, self.unfinished[sut_interaction])
            self._debug(f"wrote {sut_interaction}")
            del self.unfinished[sut_interaction]
