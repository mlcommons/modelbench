import json
from datetime import datetime
from inspect import getframeinfo, currentframe
from io import StringIO
from json import JSONDecodeError
from multiprocessing.pool import ThreadPool
from typing import Any

import pytest
from pydantic import BaseModel

from modelbench.run_journal import RunJournal, for_journal


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


class TestRunJournal:
    def test_file_output(self, tmp_path, capsys):
        journal_file = tmp_path / "journal.jsonl"
        with RunJournal(journal_file):
            pass
        lines = journal_file.read_text().splitlines()
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

    def test_thread_safety(self, tmp_path):
        journal_file = tmp_path / "journal.jsonl"
        with RunJournal(journal_file) as journal:

            def f(n):
                journal.raw_entry("thread_entry", entry=n)

            with ThreadPool(16) as pool:
                pool.map(f, range(16 * 16))

        lines = journal_file.read_text().splitlines()
        assert len(lines) == 1 + 16 * 16
        items_seen = set()
        for line in lines[1:]:
            j = json.loads(line)
            assert j["message"] == "thread_entry"
            items_seen.add(j["entry"])
        assert len(items_seen) == 16 * 16


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

    def test_pydantic(self):
        class Thingy(BaseModel):
            count: int
            text: str
            not_relevant: Any
            boring: str = "boring"

        assert for_journal(Thingy(count=1, text="foo", not_relevant=None)) == {"count": 1, "text": "foo"}

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
