import inspect
import json
import threading
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from io import IOBase
from unittest.mock import MagicMock

from pydantic import BaseModel


def for_journal(o):
    from modelbench.benchmark_runner import TestRunItem

    """Turns anything into a collection of primitives suitable for JSON rendering."""
    if isinstance(o, TestRunItem):
        # TODO: figure out what to do about these other fields
        #     annotations: dict[str, Annotation] = dataclasses.field(default_factory=dict)
        #     measurements: dict[str, float] = dataclasses.field(default_factory=dict)
        #     exception = None
        result = {"test": o.test.uid, "item": o.source_id(), "sut": o.sut.uid}
        return result
    elif isinstance(o, Exception):
        return {"class": o.__class__.__name__, "message": str(o)}
    elif isinstance(o, MagicMock):
        # to make testing easier
        return {"class": o.__class__.__name__}
    elif isinstance(o, BaseModel):
        return o.model_dump(exclude_defaults=True, exclude_none=True)
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
                entry[key] = for_journal(value)
            self._write(entry)

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
