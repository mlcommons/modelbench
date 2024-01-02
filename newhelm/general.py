from dataclasses import asdict, is_dataclass
import inspect
import hashlib
import json
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


def _asdict_without_nones(obj: Any) -> Dict[str, Any]:
    if not is_dataclass(obj):
        raise ValueError(f"Expected dataclass, got '{obj}'")
    return asdict(obj, dict_factory=lambda x: {k: v for (k, v) in x if v is not None})


def to_json(obj, indent=None) -> str:
    if not is_dataclass(obj):
        raise ValueError(f"Expected dataclass, got '{obj}'")
    return json.dumps(_asdict_without_nones(obj), indent=indent)


def from_json(cls: type[_InT], value: str) -> _InT:
    return dacite.from_dict(cls, json.loads(value), config=dacite.Config(strict=True))


def get_concrete_subclasses(cls: Type[_InT]) -> Set[Type[_InT]]:
    result = set()
    for subclass in cls.__subclasses__():
        if not inspect.isabstract(subclass):
            result.add(subclass)
        result.update(get_concrete_subclasses(subclass))
    return result


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
