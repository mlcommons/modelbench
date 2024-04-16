from collections import defaultdict
from typing import Callable, Mapping, Sequence


def group_by_key(items: Sequence, key=Callable) -> Mapping:
    """Group `items` by the value returned by `key(item)`."""
    groups = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    return groups
