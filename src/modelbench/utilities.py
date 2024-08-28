import ctypes
from collections import defaultdict
from multiprocessing import Value
from typing import Callable, Mapping, Sequence


def group_by_key(items: Sequence, key=Callable) -> Mapping:
    """Group `items` by the value returned by `key(item)`."""
    groups = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    return groups


class ProgressTracker:
    def __init__(self, total, print_updates=False, shared_count=None, lock=None):
        assert total > 0, f"Total must be a positive number, got {total}."
        self.complete_count = shared_count or Value(ctypes.c_double, 0.0)
        self.total = total
        self.print_updates = print_updates
        self.lock = lock
        if self.print_updates:
            # Print initial progress json.
            self.print_progress_json()

    @property
    def progress(self):
        return self.complete_count.value / self.total

    def print_progress_json(self):
        print({"progress": self.progress})

    def increment(self):
        if self.lock:
            with self.lock:
                self._increment()
        else:
            self._increment()

    def _increment(self):
        self.complete_count.value += 1.0
        if self.print_updates:
            self.print_progress_json()
