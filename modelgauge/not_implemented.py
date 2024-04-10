from functools import wraps


def not_implemented(f):
    """Decorate a method as not implemented in a way we can detect."""

    @wraps(f)
    def inner(*args, **kwargs):
        f(*args, **kwargs)
        # We expect the previous line to raise a NotImplementedError, assert if it doesn't
        raise AssertionError(f"Expected {f} to raise a NotImplementedError.")

    inner._not_implemented = True
    return inner


def is_not_implemented(f) -> bool:
    """Check if a method is decorated with @not_implemented."""
    return getattr(f, "_not_implemented", False)
