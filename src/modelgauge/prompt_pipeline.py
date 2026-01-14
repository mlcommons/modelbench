import time
from typing import Optional

from modelgauge.dataset import PromptDataset, PromptResponseDataset
from modelgauge.log_config import get_logger
from modelgauge.pipeline import CachingPipe, Pipe, Sink, Source
from modelgauge.prompt import TextPrompt
from modelgauge.single_turn_prompt_response import SUTInteraction, TestItem
from modelgauge.sut import PromptResponseSUT, SUT, SUTResponse
from modelgauge.sut_capabilities import AcceptsTextPrompt, ProducesPerTokenLogProbabilities
from modelgauge.sut_capabilities_verification import assert_multiple_suts_capabilities
from modelgauge.model_options import ModelOptions

logger = get_logger(__name__)


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
    def __init__(self, suts: dict[str, SUT], sut_options: Optional[ModelOptions] = None, workers=None, cache_path=None):
        self.sleep_time = 10
        if workers is None:
            workers = 8
        super().__init__(thread_count=workers, cache_path=cache_path)
        self.suts = suts
        self.sut_options = sut_options
        self.sut_response_counts = {uid: 0 for uid in suts}
        self._assert_capabilities()

    def _assert_capabilities(self):
        required_capabilities = [AcceptsTextPrompt]
        if self.sut_options and self.sut_options.top_logprobs is not None:
            required_capabilities.append(ProducesPerTokenLogProbabilities)
        assert_multiple_suts_capabilities(list(self.suts.values()), required_capabilities)

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
    def __init__(self, writer: PromptResponseDataset):
        super().__init__()
        self.writer = writer

    def run(self):
        with self.writer:
            super().run()

    def handle_item(self, item: SUTInteraction):
        self.writer.write(item)
