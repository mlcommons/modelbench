import json
import time
from datetime import datetime
from inspect import getframeinfo, currentframe
from io import StringIO, TextIOWrapper
from json import JSONDecodeError
from multiprocessing.pool import ThreadPool
from typing import Any

import pytest
import zstandard
from pydantic import BaseModel

from modelbench.benchmark_runner_items import Timer
from modelbench.run_journal import RunJournal, for_journal
from modelgauge.locales import EN_US
from modelgauge.sut import SUTResponse
from modelgauge.model_options import TokenProbability, TopTokens


def assert_no_output(capsys):
    out, err = capsys.readouterr()
    assert not (out or err), f"out = {out}, err = {err}"


class FakeOutput(StringIO):

    def close(self):
        pass

    def dispose(self):
        super().close()

    def lines(self):
        return self.getvalue().splitlines()

    def entry(self, position):
        line = self.lines()[position]
        try:
            return json.loads(line)
        except JSONDecodeError:
            raise ValueError(f"Failed to decode {line}")

    def last_entry(self):
        return self.entry(-1)


class FakeJournal(FakeOutput, RunJournal):
    def __init__(self):
        FakeOutput.__init__(self)
        RunJournal.__init__(self, self)


@pytest.fixture()
def journal() -> FakeJournal:
    journal = FakeJournal()
    with journal:
        yield journal
    journal.dispose()


class TestForJournal:
    def test_primitives(self):
        assert for_journal(None) is None
        assert for_journal(1) is 1
        assert for_journal(1.1) == 1.1
        assert for_journal("one") is "one"

    def test_list(self):
        assert for_journal(["one", "two"]) == ["one", "two"]

    def test_dict(self):
        assert for_journal({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_locale(self):
        assert for_journal(EN_US) == EN_US

    def test_nested_objects(self):
        assert for_journal([EN_US]) == [EN_US]
        assert for_journal({"locale": EN_US}) == {"locale": EN_US}
        assert for_journal({"a_list": [EN_US]}) == {"a_list": [EN_US]}

    def test_pydantic(self):
        class Thingy(BaseModel):
            count: int
            text: str
            not_relevant: Any
            boring: str = "boring"

        assert for_journal(Thingy(count=1, text="foo", not_relevant=None)) == {"count": 1, "text": "foo"}

    def test_sut_response(self):
        no_logprobs = SUTResponse(text="foo")
        assert for_journal(no_logprobs) == {"response_text": "foo"}

        # the logprobs seem wildly over-nested to me, but I'm not sure, so I'm leaving them as is
        with_logprobs = SUTResponse(
            text="foo", top_logprobs=[TopTokens(top_tokens=[TokenProbability(token="f", logprob=1.0)])]
        )
        logprob_result = for_journal(with_logprobs)
        assert logprob_result["response_text"] == "foo"
        logprobs = logprob_result["logprobs"][0]["top_tokens"][0]
        assert logprobs["token"] == "f"
        assert logprobs["logprob"] == 1.0

    def test_exception(self):
        f = getframeinfo(currentframe())
        try:
            x = 1 / 0
        except ZeroDivisionError as e:
            j = for_journal(e)
            assert j["class"] == "ZeroDivisionError"
            assert j["message"] == "division by zero"
            assert j["filename"] == __file__
            assert j["lineno"] == f.lineno + 2
            assert j["function"] == "test_exception"
            assert j["arguments"] == {"self": repr(self)}
            assert j["variables"] == {"f": repr(f), "e": repr(e)}

    def test_timer(self):
        with Timer() as t:
            time.sleep(0.001)

        assert for_journal(t) == pytest.approx(0.001, 4)


def reader_for(path):
    if path.suffix == ".zst":
        raw_fh = open(path, "rb")
        dctx = zstandard.ZstdDecompressor()
        sr = dctx.stream_reader(raw_fh)
        return TextIOWrapper(sr, encoding="utf-8")
    else:
        return open(path, "r")


class TestRunJournal:
    def test_file_output(self, tmp_path, capsys):
        journal_file = tmp_path / "journal.jsonl.zst"
        with RunJournal(journal_file):
            pass
        lines = reader_for(journal_file).readlines()
        assert len(lines) == 1
        assert_no_output(capsys)

    def test_filehandle_output(self, capsys):
        o = FakeOutput()
        with RunJournal(o):
            pass
        lines = o.lines()
        assert len(lines) == 1
        assert_no_output(capsys)

    def test_no_output(self, capsys):
        with RunJournal():
            pass
        assert_no_output(capsys)

    def test_message_format(self, journal):
        journal.raw_entry("scratch")
        e = journal.last_entry()
        assert e["message"] == "scratch"
        assert datetime.fromisoformat(e["timestamp"]).year >= 2024

    def test_message_with_standard_kwargs(self, journal):
        journal.raw_entry("scratch", flavor="vanilla", cone="waffle", scoops=3, sprinkles=True)
        e = journal.last_entry()
        assert e["flavor"] == "vanilla"
        assert e["cone"] == "waffle"
        assert e["scoops"] == 3
        assert e["sprinkles"] == True

    def test_class_and_method_normal(self, journal):
        journal.raw_entry("scratch", foo="bar")

        e = journal.entry(0)
        assert e["class"] == FakeJournal.__name__
        assert e["method"] == "__init__"

        e = journal.entry(-1)
        assert e["class"] == self.__class__.__name__
        assert e["method"] == "test_class_and_method_normal"

    def test_exception_output(self, journal):
        journal.raw_entry("exception", exception=ValueError("your values are suspicious"))

        e = journal.last_entry()
        assert e["message"] == "exception"
        assert e["exception"]["class"] == "ValueError"
        assert e["exception"]["message"] == "your values are suspicious"

    def test_run_item_output(self, journal):
        test_run_item = self.make_test_run_item("id1", "a_test", "Hello?")

        journal.item_entry("an item", test_run_item)

        e = journal.last_entry()
        assert e["message"] == "an item"
        assert e["test"] == "a_test"
        assert e["prompt_id"] == "id1"

    def test_run_item_output_with_sut(self, journal, sut):
        tri = self.make_test_run_item("id1", "a_test", "Hello?")
        tri.sut = sut

        journal.item_entry("an item", tri)

        e = journal.last_entry()
        assert e["sut"] == tri.sut.uid

    def test_run_item_output_with_extra_args(self, journal):
        tri = self.make_test_run_item("id1", "a_test", "Hello?")

        journal.item_entry("an item", tri, one=1, two=2)

        e = journal.last_entry()
        assert e["one"] == 1
        assert e["two"] == 2

    def test_item_exception_entry(self, journal, sut):
        tri = self.make_test_run_item("id1", "a_test", "Hello?")
        tri.sut = sut

        journal.item_exception_entry("fail", tri, ValueError())

        e = journal.last_entry()
        assert e["message"] == "fail"
        assert e["test"] == "a_test"
        assert e["prompt_id"] == "id1"
        assert e["sut"] == tri.sut.uid
        assert e["exception"]["class"] == "ValueError"

    def make_test_run_item(self, source_id, test_id, text):
        from modelbench.benchmark_runner import TestRunItem, ModelgaugeTestWrapper
        from modelbench_tests.test_benchmark_runner import AFakeTest
        from modelgauge.prompt import TextPrompt
        from modelgauge.single_turn_prompt_response import TestItem

        test_item = TestItem(prompt=TextPrompt(text=text), source_id=source_id)
        test = ModelgaugeTestWrapper(AFakeTest(test_id, [test_item]), None)
        test_run_item = TestRunItem(test, test_item)
        return test_run_item

    def test_thread_safety(self, tmp_path):
        journal_file = tmp_path / "journal.jsonl.zst"
        with RunJournal(journal_file) as journal:

            def f(n):
                journal.raw_entry("thread_entry", entry=n)

            with ThreadPool(16) as pool:
                pool.map(f, range(16 * 16))

        lines = reader_for(journal_file).readlines()
        assert len(lines) == 1 + 16 * 16
        items_seen = set()
        for line in lines[1:]:
            j = json.loads(line)
            assert j["message"] == "thread_entry"
            items_seen.add(j["entry"])
        assert len(items_seen) == 16 * 16
