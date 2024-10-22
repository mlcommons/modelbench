import json
from datetime import datetime
from io import StringIO
from json import JSONDecodeError

import pytest

from modelbench.run_journal import RunJournal


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


class FakeJournal(RunJournal):
    def __init__(self):
        self.output = FakeOutput()
        super().__init__(self.output)

    def __getattr__(self, name):
        return getattr(self.output, name)


@pytest.fixture()
def journal():
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
