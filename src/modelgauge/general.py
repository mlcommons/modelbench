import datetime
import hashlib
import importlib
import inspect
import logging
import shlex
import subprocess
import time
from typing import List, Optional, Set, Type, TypeVar

from tqdm import tqdm

# Type vars helpful in defining templates.
_InT = TypeVar("_InT")


def current_timestamp_millis() -> int:
    return time.time_ns() // 1_000_000


def get_concrete_subclasses(cls: Type[_InT]) -> Set[Type[_InT]]:
    result = set()
    for subclass in cls.__subclasses__():
        if not inspect.isabstract(subclass):
            result.add(subclass)
        result.update(get_concrete_subclasses(subclass))
    return result


def value_or_default(value: Optional[_InT], default: _InT) -> _InT:
    if value is not None:
        return value
    return default


def shell(args: List[str]):
    """Executes the shell command in `args`."""
    cmd = shlex.join(args)
    logging.info(f"Executing: {cmd}")
    exit_code = subprocess.call(args)
    if exit_code != 0:
        logging.error(f"Failed with exit code {exit_code}: {cmd}")


def hash_file(filename, block_size=65536):
    """Apply sha256 to the bytes of `filename`."""
    file_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            file_hash.update(block)

    return file_hash.hexdigest()


def normalize_filename(filename: str) -> str:
    """Replace filesystem characters in `filename`."""
    return filename.replace("/", "_")


class UrlRetrieveProgressBar:
    """Progress bar compatible with urllib.request.urlretrieve."""

    def __init__(self, url: str):
        self.bar = None
        self.url = url

    def __call__(self, block_num, block_size, total_size):
        if not self.bar:
            self.bar = tqdm(total=total_size, unit="B", unit_scale=True)
            self.bar.set_description(f"Downloading {self.url}")
        self.bar.update(block_size)


def get_class(module_name: str, qual_name: str):
    """Get the class object given its __module__ and __qualname__."""
    scope = importlib.import_module(module_name)
    names = qual_name.split(".")
    for name in names:
        scope = getattr(scope, name)
    return scope


def current_local_datetime():
    """Get the current local date time, with timezone."""
    return datetime.datetime.now().astimezone()


class APIException(Exception):
    """Failure in or with an underlying API. Consider specializing for
    specific errors that should be handled differently."""


class TestItemError(Exception):
    """Error encountered while processing a test item"""
