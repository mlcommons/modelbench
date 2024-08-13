from collections import defaultdict
from typing import Callable, Mapping, Sequence


def group_by_key(items: Sequence, key=Callable) -> Mapping:
    """Group `items` by the value returned by `key(item)`."""
    groups = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    return groups


class ProgressTracker:
    def __init__(self, total, print_updates=False):
        self.complete_count = 0.0
        self.total = total
        self.print_updates = print_updates

    def increment(self):
        self.complete_count += 1.0
        if self.print_updates:
            print({"progress": self.complete_count / self.total})
