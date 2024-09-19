import importlib
from modelgauge.dependency_injection import (
    inject_dependencies,
    serialize_injected_dependencies,
)
from modelgauge.secret_values import RawSecrets
from pydantic import BaseModel
from typing import Any, List, Mapping


class InitializationRecord(BaseModel):
    """Holds data sufficient to reconstruct an object."""

    module: str
    class_name: str
    args: List[Any]
    kwargs: Mapping[str, Any]

    def recreate_object(self, *, secrets: RawSecrets = {}):
        """Redoes the init call from this record."""
        cls = getattr(importlib.import_module(self.module), self.class_name)
        args, kwargs = inject_dependencies(self.args, self.kwargs, secrets=secrets)
        return cls(*args, **kwargs)


def add_initialization_record(self, *args, **kwargs):
    record_args, record_kwargs = serialize_injected_dependencies(args, kwargs)
    self.initialization_record = InitializationRecord(
        module=self.__class__.__module__,
        class_name=self.__class__.__qualname__,
        args=record_args,
        kwargs=record_kwargs,
    )
