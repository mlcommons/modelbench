import inspect
import json
import threading
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from enum import Enum
from io import IOBase, TextIOWrapper
from typing import Sequence, Mapping
from unittest.mock import MagicMock

from pydantic import BaseModel
from zstandard.backend_cffi import ZstdCompressor, ZstdDecompressor

from modelbench.benchmark_runner_items import TestRunItem, Timer
from modelgauge.sut import SUTResponse


def for_journal(o):
    """Turns anything into a collection of primitives suitable for JSON rendering."""
    if isinstance(o, TestRunItem):
        return {"test": o.test.uid, "item": o.source_id(), "sut": o.sut.uid}
    if isinstance(o, SUTResponse):
        result = {"response_text": o.text}
        if o.top_logprobs is not None:
            result["logprobs"] = for_journal(o.top_logprobs)
        return result
    elif isinstance(o, BaseModel):
        return for_journal(o.model_dump(exclude_defaults=True, exclude_none=True))
    elif isinstance(o, Sequence) and not isinstance(o, str):
        return [for_journal(i) for i in o]
    elif isinstance(o, Mapping):
        return {k: for_journal(v) for k, v in o.items()}
    elif isinstance(o, Enum):
        return o.value
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


def journal_reader(path):
    """Loads existing journal file, decompressing if necessary."""
    if path.suffix == ".zst":
        raw_fh = open(path, "rb")
        dctx = ZstdDecompressor()
        sr = dctx.stream_reader(raw_fh)
        return TextIOWrapper(sr, encoding="utf-8")
    else:
        return open(path, "r")


class RunJournal(AbstractContextManager):

    def __init__(self, output=None):
        super().__init__()
        if isinstance(output, IOBase):
            self.filehandle = output
            self.binary = False
        elif output:
            out = open(str(output), "wb")
            cctx = ZstdCompressor()
            self.filehandle = cctx.stream_writer(out)
            self.binary = True
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
            entry["prompt_id"] = item.source_id()
        except Exception as e:
            # another good place for logging
            entry["prompt_id"] = f"failed: {e}"
        if item.sut:
            entry["sut"] = item.sut.uid
        return entry

    def _write(self, entry):
        with self.output_lock:
            j = json.dumps(entry) + "\n"
            if self.binary:
                self.filehandle.write(j.encode("utf-8"))
            else:
                self.filehandle.write(j)

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
