import inspect
import json
import threading
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from io import IOBase


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
            entry = {"message": message, "timestamp": self._timestamp()}
            entry.update(self._caller_info())
            for key, value in kwargs.items():
                entry[key] = value
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
