import csv
import logging
import time
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Iterable, Optional

from modelgauge.dataset import PromptDataset, PromptResponseDataset
from modelgauge.data_schema import DEFAULT_PROMPT_RESPONSE_SCHEMA, DEFAULT_PROMPT_SCHEMA, PromptSchema
from modelgauge.pipeline import CachingPipe, Pipe, Sink, Source
from modelgauge.prompt import TextPrompt
from modelgauge.single_turn_prompt_response import SUTInteraction, TestItem
from modelgauge.sut import PromptResponseSUT, SUT, SUTOptions, SUTResponse

logger = logging.getLogger(__name__)


class PromptOutput(metaclass=ABCMeta):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abstractmethod
    def write(self, item, results):
        pass


class CsvPromptOutput(PromptOutput):
    """Outputs a CSV file where each row represents one SUT's response to a prompt."""

    schema = DEFAULT_PROMPT_RESPONSE_SCHEMA

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
        self.writer.writerow(
            [self.schema.prompt_uid, self.schema.prompt_text, self.schema.sut_uid, self.schema.sut_response]
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def write(self, item: TestItem, results):
        base_row = [item.source_id, item.prompt.text]  # type: ignore
        for sut in self.suts:
            if sut in results:
                self.writer.writerow(base_row + [sut, results[sut]])


class PromptSource(Source):
    def __init__(self, input: PromptDataset):
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
        self.sleep_time = 10
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
        return SUTInteraction(prompt_item, sut_uid, response)

    def call_sut(self, prompt_text: TextPrompt, sut: PromptResponseSUT) -> SUTResponse:
        request = sut.translate_text_prompt(prompt_text, self.sut_options)
        tries = 0
        while True:
            tries += 1
            try:
                response = sut.evaluate(request)
                break
            except Exception as e:
                logger.warning(f"Exception calling SUT {sut.uid} on attempt {tries}: {e}\nRetrying.....", exc_info=True)
                time.sleep(self.sleep_time)
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

    def handle_item(self, item: SUTInteraction):
        self.unfinished[item.prompt][item.sut_uid] = item.response.text
        if len(self.unfinished[item.prompt]) == len(self.suts):
            self.writer.write(item.prompt, self.unfinished[item.prompt])
            self._debug(f"wrote {item.prompt}")
            del self.unfinished[item.prompt]
