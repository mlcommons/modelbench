from functools import wraps
import importlib
from typing import Any, List, Mapping
from pydantic import BaseModel


class InitializationRecord(BaseModel):
    """Holds data sufficient to reconstruct an object."""

    module: str
    qual_name: str
    args: List[Any]
    kwargs: Mapping[str, Any]

    def recreate_object(self):
        """Redoes the init call from this record."""
        cls = getattr(importlib.import_module(self.module), self.qual_name)
        return cls(*self.args, **self.kwargs)


def record_init(init):
    """Decorator for the __init__ function to store what arguments were passed."""

    @wraps(init)
    def wrapped_init(*args, **kwargs):
        self, real_args = args[0], args[1:]
        # We want the outer-most init to be recorded, so don't overwrite it.
        if not hasattr(self, "_initialization_record"):
            self._initialization_record = InitializationRecord(
                module=self.__class__.__module__,
                qual_name=self.__class__.__qualname__,
                args=real_args,
                kwargs=kwargs,
            )
        init(*args, **kwargs)

    return wrapped_init


def get_initialization_record(obj) -> InitializationRecord:
    """Get the initialization record from an object."""
    try:
        return obj._initialization_record
    except AttributeError:
        raise AssertionError(
            f"Class {obj.__class__.__qualname__} in module "
            f"{obj.__class__.__module__} needs to add "
            f"`@record_init` to its `__init__` function to "
            f"enable system reproducibility."
        )
