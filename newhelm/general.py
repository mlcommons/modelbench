from dataclasses import asdict, is_dataclass
import inspect
import hashlib
import json
import os
import shlex
import subprocess
import time
from typing import Any, Dict, List, Set, Type, TypeVar
import uuid

import dacite

# Type vars helpful in defining templates.
_InT = TypeVar("_InT")


def get_unique_id() -> str:
    return uuid.uuid4().hex


def current_timestamp_millis() -> int:
    return time.time_ns() // 1_000_000


def asdict_without_nones(obj: Any) -> Dict[str, Any]:
    if not is_dataclass(obj):
        raise ValueError(f"Expected dataclass, got '{obj}'")
    return asdict(obj, dict_factory=lambda x: {k: v for (k, v) in x if v is not None})


def to_json(obj, indent=None) -> str:
    if not is_dataclass(obj):
        raise ValueError(f"Expected dataclass, got '{obj}'")
    return json.dumps(asdict_without_nones(obj), indent=indent)


def from_dict(cls: type[_InT], dict: Dict) -> _InT:
    return dacite.from_dict(cls, dict, config=dacite.Config(strict=True))


def from_json(cls: type[_InT], value: str) -> _InT:
    return from_dict(cls, json.loads(value))


def get_concrete_subclasses(cls: Type[_InT]) -> Set[Type[_InT]]:
    result = set()
    for subclass in cls.__subclasses__():
        if not inspect.isabstract(subclass):
            result.add(subclass)
        result.update(get_concrete_subclasses(subclass))
    return result


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
    print(f"Executing: {cmd}")
    exit_code = subprocess.call(args)
    if exit_code != 0:
        print(f"Failed with exit code {exit_code}: {cmd}")


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
