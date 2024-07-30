import csv
from abc import abstractmethod, ABCMeta
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from modelgauge.pipeline import Source, Pipe, Sink, CachingPipe
from modelgauge.prompt import TextPrompt
from modelgauge.single_turn_prompt_response import PromptWithContext
from modelgauge.sut import PromptResponseSUT, SUT, SUTCompletion


PROMPT_CSV_INPUT_COLUMNS = ["UID", "Text"]


@dataclass
class SutInteraction:
    prompt: PromptWithContext
    sut_uid: str
    response: SUTCompletion

    def __hash__(self):
        return hash(self.prompt.source_id + self.sut_uid)


class PromptInput(metaclass=ABCMeta):
    """
    Your subclass should implement __iter__ such that it yields PromptWithContext objects.
    Note that the source_id field must be set.
    """

    @abstractmethod
    def __iter__(self) -> Iterable[PromptWithContext]:
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
        self._validate_file()

    def __iter__(self) -> Iterable[PromptWithContext]:
        with open(self.path, newline="") as f:
            csvreader = csv.DictReader(f)
            for row in csvreader:
                yield PromptWithContext(
                    prompt=TextPrompt(text=row["Text"]),
                    # Forward the underlying id to help make data tracking easier.
                    source_id=row["UID"],
                    # Context can be any type you want.
                    context=row,
                )

    def _validate_file(self):
        with open(self.path, newline="") as f:
            csvreader = csv.reader(f)
            columns = next(csvreader)
        assert all(
            c in columns for c in PROMPT_CSV_INPUT_COLUMNS
        ), f"Invalid input file. Must have columns: {', '.join(PROMPT_CSV_INPUT_COLUMNS)}."


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
        assert (
            path.suffix.lower() == ".csv"
        ), f"Invalid output file {path}. Must be of type CSV."

        self.path = path
        self.suts = suts
        self.file = None
        self.writer = None

    def __enter__(self):
        self.file = open(self.path, "w", newline="")
        self.writer = csv.writer(self.file, quoting=csv.QUOTE_ALL)
        self.writer.writerow(PROMPT_CSV_INPUT_COLUMNS + [s for s in self.suts.keys()])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def write(self, item: PromptWithContext, results):
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
    def __init__(self, suts: dict[str, SUT], workers=None, cache_path=None):
        if workers is None:
            workers = 8
        super().__init__(thread_count=workers, cache_path=cache_path)
        self.suts = suts

    def key(self, item):
        prompt_item: PromptWithContext
        prompt_item, sut_uid = item
        return (prompt_item.source_id, prompt_item.prompt.text, sut_uid)

    def handle_uncached_item(self, item):
        prompt_item: PromptWithContext
        prompt_item, sut_uid = item
        response = self.call_sut(prompt_item.prompt, self.suts[sut_uid])
        return SutInteraction(prompt_item, sut_uid, response)

    def call_sut(
        self, prompt_text: TextPrompt, sut: PromptResponseSUT
    ) -> SUTCompletion:
        request = sut.translate_text_prompt(prompt_text)
        response = sut.evaluate(request)
        result = sut.translate_response(request, response)
        return result.completions[0]


class PromptSink(Sink):
    unfinished: defaultdict[PromptWithContext, dict[str, str]]

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
