import itertools
import signal
import time
from csv import DictReader
from typing import List
from unittest.mock import MagicMock

import pytest

from modelgauge.dataset import PromptDataset, PromptResponseDataset
from modelgauge.data_schema import PromptResponseSchema, PromptSchema, SchemaValidationError
from modelgauge.pipeline import Pipeline, PipelineSegment
from modelgauge.prompt import TextPrompt
from modelgauge.prompt_pipeline import (
    PromptSink,
    PromptSource,
    PromptSutAssigner,
    PromptSutWorkers,
)
from modelgauge.single_turn_prompt_response import SUTInteraction, TestItem
from modelgauge.sut import SUTResponse
from modelgauge.model_options import ModelOptions

from modelgauge_tests.fake_sut import FakeSUT, FakeSUTRequest, FakeSUTResponse

PROMPT_SCHEMA = PromptSchema.default()
PROMPT_RESPONSE_SCHEMA = PromptResponseSchema.default()


class timeout:
    def __init__(self, seconds: int):
        self.seconds = seconds

    def handle_timeout(self, signum, frame):
        raise TimeoutError(f"took more than {self.seconds}s to run")

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


class FakePromptInput:
    def __init__(self, items: list[dict], delay=None):
        super().__init__()
        self.items = items
        self.delay = itertools.cycle(delay or [0])

    def __iter__(self):
        for row in self.items:
            time.sleep(next(self.delay))
            yield TestItem(
                prompt=TextPrompt(text=row[PROMPT_SCHEMA.prompt_text]),
                source_id=row[PROMPT_SCHEMA.prompt_uid],
                context=row,
            )


class FakePromptOutput(PromptResponseDataset):
    def __init__(self, path: str):
        self.output = []
        super().__init__(path, "w")

    def write(self, item):
        self.output.append(item)


class FakeSUTWithDelay(FakeSUT):
    def __init__(self, uid: str = "fake-sut", delay=None):
        self.delay = itertools.cycle(delay or [0])
        super().__init__(uid)

    def evaluate(self, request: FakeSUTRequest) -> FakeSUTResponse:
        time.sleep(next(self.delay))
        return super().evaluate(request)


@pytest.fixture
def suts():
    suts = {"fake1": FakeSUT("fake1"), "fake2": FakeSUT("fake2")}
    return suts


def test_csv_prompt_input(tmp_path):
    file_path = tmp_path / "input.csv"
    file_path.write_text(f'{PROMPT_SCHEMA.prompt_uid},{PROMPT_SCHEMA.prompt_text}\n"1","a"')
    input = PromptDataset(file_path)

    assert len(input) == 1
    items: List[TestItem] = [i for i in input]
    assert items[0].source_id == "1"
    assert items[0].prompt.text == "a"
    assert len(items) == 1


@pytest.mark.parametrize("header", ["UID,Extra,Response\n", "Hello,World,Extra\n"])
def test_csv_prompt_input_invalid_columns(tmp_path, header):
    file_path = tmp_path / "input.csv"
    file_path.write_text(header)
    with pytest.raises(SchemaValidationError):
        PromptDataset(file_path)


def test_csv_prompt_output(tmp_path, suts):
    file_path = tmp_path / "output.csv"

    with PromptResponseDataset(file_path, "w") as output:
        test_item = TestItem(source_id="1", prompt=TextPrompt(text="a"))
        sut_interaction = SUTInteraction(test_item, "fake1", SUTResponse(text="a1"))
        output.write(sut_interaction)

    with open(file_path, "r", newline="") as f:
        # noinspection PyTypeChecker
        items: list[dict] = [i for i in (DictReader(f))]
        assert len(items) == 1
        assert items[0][PROMPT_RESPONSE_SCHEMA.prompt_uid] == "1"
        assert items[0][PROMPT_RESPONSE_SCHEMA.prompt_text] == "a"
        assert items[0][PROMPT_RESPONSE_SCHEMA.sut_uid] == "fake1"
        assert items[0][PROMPT_RESPONSE_SCHEMA.sut_response] == "a1"


def test_prompt_sut_worker_normal(suts):
    mock = MagicMock()
    mock.return_value = FakeSUTResponse(text="a response")
    suts["fake1"].evaluate = mock
    prompt_with_context = TestItem(source_id="1", prompt=TextPrompt(text="a prompt"))

    w = PromptSutWorkers(suts)
    result = w.handle_item((prompt_with_context, "fake1"))

    assert result == SUTInteraction(prompt_with_context, "fake1", SUTResponse(text="a response"))


def test_prompt_sut_worker_sends_prompt_options(suts):
    mock = MagicMock()
    mock.return_value = FakeSUTRequest(text="")
    suts["fake1"].translate_text_prompt = mock
    prompt = TextPrompt(text="a prompt")
    sut_options = ModelOptions(max_tokens=42, top_p=0.5, temperature=0.5)
    prompt_with_context = TestItem(source_id="1", prompt=prompt)

    w = PromptSutWorkers(suts, sut_options=sut_options)
    w.handle_item((prompt_with_context, "fake1"))

    mock.assert_called_with(prompt, sut_options)


def test_prompt_sut_worker_cache(suts, tmp_path):
    mock = MagicMock()
    mock.return_value = FakeSUTResponse(text="a response")
    suts["fake1"].evaluate = mock
    prompt_with_context = TestItem(source_id="1", prompt=TextPrompt(text="a prompt"))

    w = PromptSutWorkers(suts, cache_path=tmp_path)
    result = w.handle_item((prompt_with_context, "fake1"))
    assert result == SUTInteraction(prompt_with_context, "fake1", SUTResponse(text="a response"))
    assert mock.call_count == 1

    result = w.handle_item((prompt_with_context, "fake1"))
    assert result == SUTInteraction(prompt_with_context, "fake1", SUTResponse(text="a response"))
    assert mock.call_count == 1


def test_prompt_sut_worker_retries_until_success(suts):
    num_exceptions = 3
    mock = MagicMock()
    exceptions = [Exception() for _ in range(num_exceptions)]
    mock.side_effect = exceptions + [FakeSUTResponse(text="a response")]
    suts["fake1"].evaluate = mock
    prompt_with_context = TestItem(source_id="1", prompt=TextPrompt(text="a prompt"))

    w = PromptSutWorkers(suts)
    w.sleep_time = 0.01
    result = w.handle_item((prompt_with_context, "fake1"))
    assert result == SUTInteraction(prompt_with_context, "fake1", SUTResponse(text="a response"))
    assert mock.call_count == num_exceptions + 1


def test_full_run(suts, tmp_path):
    input = FakePromptInput(
        [
            {PROMPT_SCHEMA.prompt_uid: "1", PROMPT_SCHEMA.prompt_text: "a"},
            {PROMPT_SCHEMA.prompt_uid: "2", PROMPT_SCHEMA.prompt_text: "b"},
        ]
    )
    output = FakePromptOutput(tmp_path / "output.csv")

    p = Pipeline(
        PromptSource(input),
        PromptSutAssigner(suts),
        PromptSutWorkers(suts, workers=1),
        PromptSink(output),
        debug=True,
    )

    p.run()

    assert len(output.output) == len(input.items) * len(suts)  # One row per prompt per SUT
    # Every sut uid and prompt uid should be present
    assert set(row.sut_uid for row in output.output) == set(suts.keys())
    assert set(row.prompt.source_id for row in output.output) == {"1", "2"}


@pytest.mark.parametrize("worker_count", [1, 2, 4, 8])
def test_concurrency_with_delays(suts, worker_count, tmp_path):
    PipelineSegment.default_timeout = 0.001  # burn some CPU to make the tests run faster

    prompt_count = worker_count * 4
    prompt_delays = [0, 0.01, 0.02]
    sut_delays = [0, 0.01, 0.02, 0.03]
    suts = {
        "fake1": FakeSUTWithDelay("fake1", delay=sut_delays),
        "fake2": FakeSUTWithDelay("fake2", delay=sut_delays),
    }
    input = FakePromptInput(
        [{PROMPT_SCHEMA.prompt_uid: str(i), PROMPT_SCHEMA.prompt_text: "text" + str(i)} for i in range(prompt_count)],
        delay=prompt_delays,
    )
    output = FakePromptOutput(tmp_path / "output.csv")

    p = Pipeline(
        PromptSource(input),
        PromptSutAssigner(suts),
        PromptSutWorkers(suts, workers=worker_count),
        PromptSink(output),
    )

    average_delay_per_prompt = sum(sut_delays) / len(sut_delays) + sum(prompt_delays) / len(sut_delays)

    with timeout(5 + int(prompt_count * average_delay_per_prompt / worker_count)):
        p.run()

    assert len(output.output) == len(input.items) * len(suts)


def test_progress(suts, tmp_path):
    input = FakePromptInput(
        [
            {PROMPT_SCHEMA.prompt_uid: "1", PROMPT_SCHEMA.prompt_text: "a"},
            {PROMPT_SCHEMA.prompt_uid: "2", PROMPT_SCHEMA.prompt_text: "b"},
        ]
    )
    output = FakePromptOutput(tmp_path / "output.csv")

    def track_progress(data):
        progress_items.append(data.copy())

    p = Pipeline(
        PromptSource(input),
        PromptSutAssigner(suts),
        PromptSutWorkers(suts, workers=2),
        PromptSink(output),
        progress_callback=track_progress,
    )
    progress_items = []

    p.run()

    assert progress_items[0]["completed"] == 0
    assert progress_items[-1]["completed"] == 4
