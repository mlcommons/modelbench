from functools import wraps
import inspect
from newhelm.base_test import BaseTest
from newhelm.record_init import add_initialization_record


def newhelm_test():
    """Decorator providing common behavior and hooks for all NewHELM Tests."""

    def inner(cls):
        assert issubclass(
            cls, BaseTest
        ), "Decorator can only be applied to classes that inherit from BaseTest."
        cls.__init__ = _wrap_init(cls.__init__)
        cls._newhelm_test = True
        return cls

    return inner


def _wrap_init(init):
    """Wrap the SUT __init__ function to verify it behaves as expected."""

    if hasattr(init, "_newhelm_wrapped"):
        # Already wrapped, no need to do any work.
        return init

    _validate_init_signature(init)

    @wraps(init)
    def wrapped_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        add_initialization_record(self, *args, **kwargs)

    wrapped_init._newhelm_wrapped = True
    return wrapped_init


def _validate_init_signature(init):
    params = list(inspect.signature(init).parameters.values())
    assert params[1].name == "uid", "All Tests must have UID as the first parameter."
