import inspect
import json
import threading
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from io import IOBase
from typing import Sequence
from unittest.mock import MagicMock

from pydantic import BaseModel

from modelbench.benchmark_runner_items import TestRunItem, Timer
from modelgauge.sut import SUTResponse


def for_journal(o):
    """Turns anything into a collection of primitives suitable for JSON rendering."""
    if isinstance(o, TestRunItem):
        return {"test": o.test.uid, "item": o.source_id(), "sut": o.sut.uid}
    if isinstance(o, SUTResponse):
        completion = o.completions[0]
        result = {"text": completion.text}
        if completion.top_logprobs is not None:
            result["logprobs"] = for_journal(completion.top_logprobs)
        return result
    elif isinstance(o, BaseModel):
        return o.model_dump(exclude_defaults=True, exclude_none=True)
    elif isinstance(o, Sequence) and not isinstance(o, str):
        return [for_journal(i) for i in o]
    elif isinstance(o, Timer):
        return o.elapsed
    elif isinstance(o, Exception):
        result = {"class": o.__class__.__name__, "message": str(o)}
        tb = o.__traceback__
        if tb is not None:
            result["lineno"] = tb.tb_lineno
            frame = tb.tb_frame
            result["filename"] = frame.f_code.co_filename
            result["function"] = frame.f_code.co_name
            argnames = {frame.f_code.co_varnames[i] for i in range(frame.f_code.co_argcount)}
            result["arguments"] = {a: repr(frame.f_locals[a]) for a in argnames}
            result["variables"] = {k: repr(frame.f_locals[k]) for k in frame.f_locals.keys() - argnames}
        return result
    elif isinstance(o, MagicMock):
        # to make testing easier
        return {"mock": str(o)}
    else:
        return o


class RunJournal(AbstractContextManager):

    def __init__(self, output=None):
        super().__init__()
        if isinstance(output, IOBase):
            self.filehandle = output
        elif output:
            self.filehandle = open(output, "w")
        else:
            self.filehandle = None
        self.output_lock = threading.Lock()

        self.raw_entry("starting journal")

    def raw_entry(self, message, **kwargs):
        if self.filehandle:
            entry = {"timestamp": self._timestamp(), "message": message}
            entry.update(self._caller_info())
            for key, value in kwargs.items():
                if isinstance(value, SUTResponse):
                    entry.update(for_journal(value))
                else:
                    entry[key] = for_journal(value)
            self._write(entry)

    def item_entry(self, message, item: TestRunItem, **kwargs):
        entry = self._item_fields(item)
        entry.update(**kwargs)
        self.raw_entry(message, **entry)

    def item_exception_entry(self, message, item: TestRunItem, exception: Exception, **kwargs):
        entry = self._item_fields(item)
        entry["exception"] = exception
        entry.update(**kwargs)
        self.raw_entry(message, **entry)

    def _item_fields(self, item):
        entry = {}
        entry["test"] = item.test.uid
        try:
            entry["source_id"] = item.source_id()
        except Exception as e:
            # another good place for logging
            entry["source_id"] = f"failed: {e}"
        if item.sut:
            entry["sut"] = item.sut.uid
        return entry

    def _write(self, entry):
        with self.output_lock:
            json.dump(entry, self.filehandle)
            self.filehandle.write("\n")

    def close(self):
        if self.filehandle:
            self.filehandle.close()

    def _timestamp(self):
        return datetime.now(timezone.utc).isoformat()

    def __exit__(self, exc_type, exc_value, traceback, /):
        self.close()

    def _caller_info(self):
        frame = inspect.currentframe()
        info = {}
        try:
            while "self" in frame.f_locals and frame.f_locals["self"] == self and frame.f_code.co_name != "__init__":
                frame = frame.f_back
            if "self" in frame.f_locals:
                info["class"] = frame.f_locals["self"].__class__.__name__
                info["method"] = frame.f_code.co_name
            else:
                info["function"] = frame.f_code.co_name
        except KeyError:
            print(f"Unexpected failure for {frame} with locals {list(frame.f_locals.keys())}")

        return info
