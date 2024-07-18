import csv
from abc import abstractmethod, ABCMeta
from collections import defaultdict
from typing import Iterable

import diskcache  # type: ignore

from modelgauge.pipeline import Source, Pipe, Sink, CachingPipe
from modelgauge.prompt import TextPrompt
from modelgauge.single_turn_prompt_response import PromptWithContext
from modelgauge.sut import PromptResponseSUT, SUT


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
                # yield PromptItem(row)


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
        self.path = path
        self.suts = suts
        self.file = None
        self.writer = None

    def __enter__(self):
        self.file = open(self.path, "w", newline="")
        self.writer = csv.writer(self.file, quoting=csv.QUOTE_ALL)
        headers = ["UID", "Text"]
        self.writer.writerow(headers + [s for s in self.suts.keys()])
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
        response_text = self.call_sut(prompt_item.prompt, self.suts[sut_uid])
        return prompt_item, sut_uid, response_text

    def call_sut(self, prompt_text: TextPrompt, sut: PromptResponseSUT) -> str:
        request = sut.translate_text_prompt(prompt_text)
        response = sut.evaluate(request)
        result = sut.translate_response(request, response)
        return result.completions[0].text


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

    def handle_item(self, item):
        prompt_item, sut_key, response = item
        self.unfinished[prompt_item][sut_key] = response
        if len(self.unfinished[prompt_item]) == len(self.suts):
            self.writer.write(prompt_item, self.unfinished[prompt_item])
            self._debug(f"wrote {prompt_item}")
            del self.unfinished[prompt_item]
