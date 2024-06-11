from functools import wraps

from modelgauge.record_init import add_initialization_record


def record_init(init):
    """Decorator for the __init__ function to store what arguments were passed."""

    @wraps(init)
    def wrapped_init(*args, **kwargs):
        self, real_args = args[0], args[1:]
        # We want the outer-most init to be recorded, so don't overwrite it.
        if not hasattr(self, "_initialization_record"):
            add_initialization_record(self, real_args, kwargs)
        init(*args, **kwargs)

    return wrapped_init
