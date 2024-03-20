from abc import ABC


class TrackedObject(ABC):
    """Base class for objects that have a UID."""

    def __init__(self, uid):
        self.uid = uid
