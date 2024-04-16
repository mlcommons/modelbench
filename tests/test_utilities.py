from dataclasses import dataclass
from itertools import groupby

from modelbench.utilities import group_by_key


@dataclass
class SomeClass:
    my_group: int
    value: int


def test_iterables_groupby():
    # This test demonstrates that itertools.groupby requires groups to be sorted.
    group_1_item_1 = SomeClass(my_group=1, value=1)
    group_1_item_2 = SomeClass(my_group=1, value=2)
    group_2_item_1 = SomeClass(my_group=2, value=1)
    group_2_item_2 = SomeClass(my_group=2, value=2)

    items = [
        # Not sorted by group
        group_1_item_1,
        group_2_item_1,
        group_1_item_2,
        group_2_item_2,
    ]
    groups = []
    for key, values in groupby(items, key=lambda c: c.my_group):
        groups.append((key, list(values)))
    # Shows that no grouping was performed.
    assert groups == [
        (1, [group_1_item_1]),
        (2, [group_2_item_1]),
        (1, [group_1_item_2]),
        (2, [group_2_item_2]),
    ]


def test_group_by_key():
    group_1_item_1 = SomeClass(my_group=1, value=1)
    group_1_item_2 = SomeClass(my_group=1, value=2)
    group_2_item_1 = SomeClass(my_group=2, value=1)
    group_2_item_2 = SomeClass(my_group=2, value=2)

    items = [
        # Not sorted by group
        group_1_item_1,
        group_2_item_1,
        group_1_item_2,
        group_2_item_2,
    ]
    groups = []
    for key, values in group_by_key(items, key=lambda c: c.my_group).items():
        groups.append((key, list(values)))
    assert groups == [
        (1, [group_1_item_1, group_1_item_2]),
        (2, [group_2_item_1, group_2_item_2]),
    ]
