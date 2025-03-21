import csv
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

from modelgauge.pipeline import CachingPipe, Pipe, Sink, Source
from modelgauge.prompt import TextPrompt
from modelgauge.single_turn_prompt_response import TestItem
from modelgauge.sut import PromptResponseSUT, SUT, SUTOptions, SUTResponse


PROMPT_CSV_INPUT_COLUMNS = {
    "default": {"id": "UID", "text": "Text"},
    "prompt_set": {"id": "release_prompt_id", "text": "prompt_text"},  # official prompt set files
    "db": {"id": "prompt_uid", "text": "prompt_text"},  # database dumps
}


@dataclass
class SutInteraction:
    prompt: TestItem
    sut_uid: str
    response: SUTResponse

    def __hash__(self):
        return hash(self.prompt.source_id + self.sut_uid)


class PromptInput(metaclass=ABCMeta):
    """
    Your subclass should implement __iter__ such that it yields TestItem objects.
    Note that the source_id field must be set.
    """

    @abstractmethod
    def __iter__(self) -> Iterable[TestItem]:
        pass

    def __len__(self):
        count = 0
        for prompt in self:
            count += 1
        return count


class CsvPromptInput(PromptInput):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.prompt_input_type = "default"
        self._identify_input()

    def _extract_field(self, row, field_name):
        column_name = PROMPT_CSV_INPUT_COLUMNS[self.prompt_input_type][field_name]
        return row[column_name]

    def __iter__(self) -> Iterable[TestItem]:
        with open(self.path, newline="") as f:
            csvreader = csv.DictReader(f)
            for row in csvreader:
                yield TestItem(
                    prompt=TextPrompt(text=self._extract_field(row, "text")),
                    # Forward the underlying id to help make data tracking easier.
                    source_id=self._extract_field(row, "id"),
                    # Context can be any type you want.
                    context=row,
                )

    def _identify_input(self):
        with open(self.path, newline="") as f:
            csvreader = csv.reader(f)
            columns = next(csvreader)
            is_valid = False
            for prompt_input_type, column_names in PROMPT_CSV_INPUT_COLUMNS.items():
                if all(c in columns for c in column_names.values()):
                    self.prompt_input_type = prompt_input_type
                    is_valid = True
                    break
        assert (
            is_valid
        ), f"Unsupported input file. Required columns are one of: f{PROMPT_CSV_INPUT_COLUMNS.values()}\nActual columns are: f{columns}"


class PromptOutput(metaclass=ABCMeta):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abstractmethod
    def write(self, item, results):
        pass


class CsvPromptOutput(PromptOutput):
    def __init__(self, path, suts):
        super().__init__()
        assert path.suffix.lower() == ".csv", f"Invalid output file {path}. Must be of type CSV."

        self.path = path
        self.suts = suts
        self.file = None
        self.writer = None

    def __enter__(self):
        self.file = open(self.path, "w", newline="")
        self.writer = csv.writer(self.file, quoting=csv.QUOTE_ALL)
        self.writer.writerow(list(PROMPT_CSV_INPUT_COLUMNS["default"].values()) + [s for s in self.suts.keys()])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def write(self, item: TestItem, results):
        row = [item.source_id, item.prompt.text]  # type: ignore
        for k in self.suts:
            if k in results:
                row.append(results[k])
            else:
                row.append("")
        self.writer.writerow(row)

    def launder_the_type_problem(self, item) -> str:
        return item.prompt.text


class PromptSource(Source):
    def __init__(self, input: PromptInput):
        super().__init__()
        self.input = input

    def new_item_iterable(self):
        return self.input


class PromptSutAssigner(Pipe):
    def __init__(self, suts: dict[str, SUT]):
        super().__init__()
        self.suts = suts

    def handle_item(self, item):
        for sut_uid in self.suts:
            self.downstream_put((item, sut_uid))


class PromptSutWorkers(CachingPipe):
    def __init__(self, suts: dict[str, SUT], sut_options: Optional[SUTOptions] = None, workers=None, cache_path=None):
        if workers is None:
            workers = 8
        super().__init__(thread_count=workers, cache_path=cache_path)
        self.suts = suts
        self.sut_options = sut_options
        self.sut_response_counts = {uid: 0 for uid in suts}

    def key(self, item):
        prompt_item: TestItem
        prompt_item, sut_uid = item
        return (prompt_item.source_id, prompt_item.prompt.text, sut_uid, self.sut_options)

    def handle_uncached_item(self, item):
        prompt_item: TestItem
        prompt_item, sut_uid = item
        response = self.call_sut(prompt_item.prompt, self.suts[sut_uid])
        return SutInteraction(prompt_item, sut_uid, response)

    def call_sut(self, prompt_text: TextPrompt, sut: PromptResponseSUT) -> SUTResponse:
        request = sut.translate_text_prompt(prompt_text, self.sut_options)
        response = sut.evaluate(request)
        result = sut.translate_response(request, response)
        self.sut_response_counts[sut.uid] += 1
        return result


class PromptSink(Sink):
    unfinished: defaultdict[TestItem, dict[str, str]]

    def __init__(self, suts: dict[str, SUT], writer: PromptOutput):
        super().__init__()
        self.suts = suts
        self.writer = writer
        self.unfinished = defaultdict(lambda: dict())

    def run(self):
        with self.writer:
            super().run()

    def handle_item(self, item: SutInteraction):
        self.unfinished[item.prompt][item.sut_uid] = item.response.text
        if len(self.unfinished[item.prompt]) == len(self.suts):
            self.writer.write(item.prompt, self.unfinished[item.prompt])
            self._debug(f"wrote {item.prompt}")
            del self.unfinished[item.prompt]
