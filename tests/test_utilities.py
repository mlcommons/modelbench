import ctypes
import pytest
from dataclasses import dataclass
from itertools import groupby
from multiprocessing import Manager, Process

from modelbench.utilities import ProgressTracker, group_by_key


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


def test_progress_tracker(capsys):
    progress = ProgressTracker(total=4, print_updates=True)

    progress.increment()
    progress.increment()

    assert progress.complete_count.value == 2
    assert progress.progress == 0.5

    captured = capsys.readouterr()
    assert captured.out == '{"progress": 0.0}\n{"progress": 0.25}\n{"progress": 0.5}\n'


def worker(progress, num_updates):
    for _ in range(num_updates):
        progress.increment()


def test_progress_tracker_concurrency(capfd):
    with Manager() as manager:
        shared_count = manager.Value(ctypes.c_double, 0.0)
        lock = manager.Lock()
        progress = ProgressTracker(total=4, print_updates=True, shared_count=shared_count, lock=lock)
        processes = [
            Process(target=worker, args=(progress, 1)),
            Process(target=worker, args=(progress, 1)),
            Process(target=worker, args=(progress, 2)),
        ]

        for p in processes:
            p.start()
        for p in processes:
            p.join()

        assert progress.complete_count.value == 4
        assert progress.progress == 1.0

        # Machine-readable progress updates are in correct order.
        captured = capfd.readouterr()
        assert (
            captured.out
            == '{"progress": 0.0}\n{"progress": 0.25}\n{"progress": 0.5}\n{"progress": 0.75}\n{"progress": 1.0}\n'
        )


def test_progress_tracker_invalid_total():
    with pytest.raises(AssertionError) as err_info:
        progress = ProgressTracker(total=0)

        assert str(err_info.value) == "Total must be a positive number, got 0."
