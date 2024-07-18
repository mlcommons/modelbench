import csv
import jsonlines
import sys
import traceback
from abc import abstractmethod, ABCMeta
from collections import defaultdict
from dataclasses import dataclass
from pydantic import BaseModel
from typing import Iterable

from modelgauge.annotator import Annotator
from modelgauge.pipeline import Source, Pipe, Sink
from modelgauge.prompt import TextPrompt
from modelgauge.prompt_pipeline import PromptOutput
from modelgauge.single_turn_prompt_response import PromptWithContext
from modelgauge.sut import PromptResponseSUT, SUTCompletion


@dataclass
class AnnotatorInputSample:
    prompt: PromptWithContext
    sut_uid: str
    response: SUTCompletion

    def __hash__(self):
        return hash(self.prompt.source_id + self.sut_uid)


class AnnotatorInput(metaclass=ABCMeta):
    @abstractmethod
    def __iter__(self) -> Iterable[AnnotatorInputSample]:
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

    def __iter__(self) -> Iterable[AnnotatorInputSample]:
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
                yield AnnotatorInputSample(prompt, row["SUT"], response)


class JsonlAnnotatorOutput(PromptOutput):
    def __init__(self, path, annotators):
        super().__init__()
        self.path = path
        self.annotators = annotators
        self.file = None
        self.writer = None

    def __enter__(self):
        self.file = open(self.path, "w", newline="")
        self.writer = jsonlines.Writer(self.file)
        # headers = ["UID", "Prompt", "SUT", "Response"]
        # for a in self.annotators.keys():
        #     headers += [a + "_is_safe", a + "_violation_categories"]
        # self.writer.write(headers)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()
        self.file.close()

    def write(self, item: AnnotatorInputSample, results):
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

    # def launder_the_type_problem(self, item) -> str:
    #     return item.prompt.text


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

    def handle_item(self, item):
        for annotator_uid in self.annotators:
            self.downstream_put((item, annotator_uid))


class AnnotatorWorkers(Pipe):
    def __init__(self, annotators: dict[str, Annotator], workers=None):
        if workers is None:
            workers = 8
        super().__init__(thread_count=workers)
        self.annotators = annotators

    def handle_item(self, item):
        annotator_input, annotator_uid = item
        try:
            annotator = self.annotators[annotator_uid]
            request = annotator.translate_request(
                annotator_input.prompt, annotator_input.response
            )
            response = annotator.annotate(request)
            result = annotator.translate_response(request, response)
            return annotator_input, annotator_uid, result
        except Exception as e:
            print(
                f"unexpected failure processing {item} for {annotator_uid}.\n{e}\n",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)


class AnnotatorSink(Sink):
    unfinished: defaultdict[AnnotatorInputSample, dict[str, str]]

    def __init__(self, annotators: dict[str, Annotator], writer: JsonlAnnotatorOutput):
        super().__init__()
        self.annotators = annotators
        self.writer = writer
        self.unfinished = defaultdict(lambda: dict())

    def run(self):
        with self.writer:
            super().run()

    def handle_item(self, item):
        annotator_input, annotator_uid, annotation = item
        if isinstance(annotation, BaseModel):
            annotation = annotation.model_dump()
        self.unfinished[annotator_input][annotator_uid] = annotation
        if len(self.unfinished[annotator_input]) == len(self.annotators):
            self.writer.write(annotator_input, self.unfinished[annotator_input])
            self._debug(f"wrote {annotator_input}")
            del self.unfinished[annotator_input]
