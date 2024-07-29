import itertools
import signal
import time
from csv import DictReader
from typing import List
from unittest.mock import MagicMock

import pytest

from modelgauge.pipeline import PipelineSegment, Pipeline
from modelgauge.prompt import TextPrompt
from modelgauge.prompt_pipeline import (
    PromptOutput,
    PromptInput,
    CsvPromptInput,
    CsvPromptOutput,
)
from modelgauge.prompt_pipeline import (
    PromptSource,
    PromptSutAssigner,
    PromptSutWorkers,
    PromptSink,
    SutInteraction,
)
from modelgauge.sut import SUTCompletion
from modelgauge.single_turn_prompt_response import PromptWithContext
from tests.fake_sut import FakeSUT, FakeSUTRequest, FakeSUTResponse


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


class FakePromptInput(PromptInput):
    def __init__(self, items: list[dict], delay=None):
        super().__init__()
        self.items = items
        self.delay = itertools.cycle(delay or [0])

    def __iter__(self):
        for row in self.items:
            time.sleep(next(self.delay))
            yield PromptWithContext(
                prompt=TextPrompt(text=row["Text"]),
                source_id=row["UID"],
                context=row,
            )


class FakePromptOutput(PromptOutput):
    def __init__(self):
        self.output = []

    def write(self, item, results):
        self.output.append({"item": item, "results": results})


class FakeSUTWithDelay(FakeSUT):
    def __init__(self, uid: str = "fake-sut", delay=None):
        self.delay = itertools.cycle(delay or [0])
        super().__init__(uid)

    def evaluate(self, request: FakeSUTRequest) -> FakeSUTResponse:
        time.sleep(next(self.delay))
        return super().evaluate(request)


@pytest.fixture
def suts():
    suts = {"fake1": FakeSUT(), "fake2": FakeSUT()}
    return suts


def test_csv_prompt_input(tmp_path):
    file_path = tmp_path / "input.csv"
    file_path.write_text('UID,Text\n"1","a"')
    input = CsvPromptInput(file_path)

    assert len(input) == 1
    items: List[PromptWithContext] = [i for i in input]
    assert items[0].source_id == "1"
    assert items[0].prompt.text == "a"
    assert len(items) == 1


@pytest.mark.parametrize("header", ["UID,Extra,Response\n", "Hello,World,Extra\n"])
def test_csv_prompt_input_invalid_columns(tmp_path, header):
    file_path = tmp_path / "input.csv"
    file_path.write_text(header)
    with pytest.raises(
        AssertionError, match="Invalid input file. Must have columns: UID, Text."
    ):
        CsvPromptInput(file_path)


def test_csv_prompt_output(tmp_path, suts):
    file_path = tmp_path / "output.csv"

    with CsvPromptOutput(file_path, suts) as output:
        output.write(
            PromptWithContext(source_id="1", prompt=TextPrompt(text="a")),
            {"fake1": "a1", "fake2": "a2"},
        )

    with open(file_path, "r", newline="") as f:
        # noinspection PyTypeChecker
        items: list[dict] = [i for i in (DictReader(f))]
        assert len(items) == 1
        assert items[0]["UID"] == "1"
        assert items[0]["Text"] == "a"
        assert items[0]["fake1"] == "a1"
        assert items[0]["fake2"] == "a2"


@pytest.mark.parametrize("output_fname", ["output.jsonl", "output"])
def test_csv_prompt_output_invalid(tmp_path, suts, output_fname):
    file_path = tmp_path / output_fname
    with pytest.raises(
        AssertionError, match=f"Invalid output file {file_path}. Must be of type CSV."
    ):
        CsvPromptOutput(file_path, suts)


def test_prompt_sut_worker_normal(suts):
    mock = MagicMock()
    mock.return_value = FakeSUTResponse(completions=["a response"])
    suts["fake1"].evaluate = mock
    prompt_with_context = PromptWithContext(
        source_id="1", prompt=TextPrompt(text="a prompt")
    )

    w = PromptSutWorkers(suts)
    result = w.handle_item((prompt_with_context, "fake1"))

    assert result == SutInteraction(
        prompt_with_context, "fake1", SUTCompletion(text="a response")
    )


def test_prompt_sut_worker_cache(suts, tmp_path):
    mock = MagicMock()
    mock.return_value = FakeSUTResponse(completions=["a response"])
    suts["fake1"].evaluate = mock
    prompt_with_context = PromptWithContext(
        source_id="1", prompt=TextPrompt(text="a prompt")
    )

    w = PromptSutWorkers(suts, cache_path=tmp_path)
    result = w.handle_item((prompt_with_context, "fake1"))
    assert result == SutInteraction(
        prompt_with_context, "fake1", SUTCompletion(text="a response")
    )
    assert mock.call_count == 1

    result = w.handle_item((prompt_with_context, "fake1"))
    assert result == SutInteraction(
        prompt_with_context, "fake1", SUTCompletion(text="a response")
    )
    assert mock.call_count == 1


def test_full_run(suts):
    input = FakePromptInput(
        [
            {"UID": "1", "Text": "a"},
            {"UID": "2", "Text": "b"},
        ]
    )
    output = FakePromptOutput()

    p = Pipeline(
        PromptSource(input),
        PromptSutAssigner(suts),
        PromptSutWorkers(suts, workers=1),
        PromptSink(suts, output),
        debug=True,
    )

    p.run()

    assert len(output.output) == len(input.items)
    assert sorted([r["item"].source_id for r in output.output]) == [
        i["UID"] for i in input.items
    ]
    row1 = output.output[0]
    assert "fake1" in row1["results"]
    assert "fake2" in row1["results"]
    row2 = output.output[1]
    assert "fake1" in row2["results"]
    assert "fake2" in row2["results"]


@pytest.mark.parametrize("worker_count", [1, 2, 4, 8])
def test_concurrency_with_delays(suts, worker_count):
    PipelineSegment.default_timeout = (
        0.001  # burn some CPU to make the tests run faster
    )

    prompt_count = worker_count * 4
    prompt_delays = [0, 0.01, 0.02]
    sut_delays = [0, 0.01, 0.02, 0.03]
    suts = {
        "fake1": FakeSUTWithDelay(delay=sut_delays),
        "fake2": FakeSUTWithDelay(delay=sut_delays),
    }
    input = FakePromptInput(
        [{"UID": str(i), "Text": "text" + str(i)} for i in range(prompt_count)],
        delay=prompt_delays,
    )
    output = FakePromptOutput()

    p = Pipeline(
        PromptSource(input),
        PromptSutAssigner(suts),
        PromptSutWorkers(suts, workers=worker_count),
        PromptSink(suts, output),
    )

    average_delay_per_prompt = sum(sut_delays) / len(sut_delays) + sum(
        prompt_delays
    ) / len(sut_delays)

    with timeout(5 + int(prompt_count * average_delay_per_prompt / worker_count)):
        p.run()

    assert len(output.output) == len(input.items)


def test_progress(suts):
    input = FakePromptInput(
        [
            {"UID": "1", "Text": "a"},
            {"UID": "2", "Text": "b"},
        ]
    )
    output = FakePromptOutput()

    def track_progress(data):
        progress_items.append(data.copy())

    p = Pipeline(
        PromptSource(input),
        PromptSutAssigner(suts),
        PromptSutWorkers(suts, workers=2),
        PromptSink(suts, output),
        progress_callback=track_progress,
    )
    progress_items = []

    p.run()

    assert progress_items[0]["completed"] == 0
    assert progress_items[-1]["completed"] == 4
