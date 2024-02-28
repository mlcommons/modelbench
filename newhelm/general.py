from dataclasses import asdict, is_dataclass
import datetime
import importlib
import inspect
import hashlib
import json
import logging
import os
import shlex
import subprocess
import time
from typing import Any, Dict, List, Optional, Set, Type, TypeVar
import uuid

from tqdm import tqdm

# Type vars helpful in defining templates.
_InT = TypeVar("_InT")


def get_unique_id() -> str:
    return uuid.uuid4().hex


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


def subset_dict(dictionary: Dict, keys) -> Dict:
    """Return a new dictionary with only specific keys.

    If a key does not exist in `dictionary`, it is ignored.
    """
    subset = {}
    for key in keys:
        try:
            subset[key] = dictionary[key]
        except KeyError:
            pass
    return subset


def get_or_create_json_file(*path_pieces):
    """Reads a json file, creating an empty one if none exists."""
    path = os.path.join(*path_pieces)
    if not os.path.exists(path):
        result = {}
        with open(path, "w") as f:
            json.dump(result, f)
        return result
    with open(path, "r") as f:
        return json.load(f)


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


class UrlRetrieveProgressBar:
    """Progress bar compatable with urllib.request.urlretrieve."""

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
