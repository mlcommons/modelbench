from abc import ABC
from newhelm.record_init import record_init


class TrackedObject(ABC):
    """Base class for objects that have a UID."""

    @record_init
    def __init__(self, uid):
        self.uid = uid
