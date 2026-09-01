import inspect
from abc import ABC


class VersionedObject(ABC):
    """Mixin requiring concrete subclasses to define a non-empty VERSION string."""

    VERSION: str

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        # this forces subclasses to have their own version
        if not cls.__dict__.get("VERSION"):
            raise TypeError(f"{cls.__name__} must define its own non-empty VERSION string.")
