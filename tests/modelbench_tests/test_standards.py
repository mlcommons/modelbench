import pathlib

import pytest

from modelbench.hazards import Standards


@pytest.fixture
def standard():
    s = Standards(pathlib.Path(__file__).parent / "data" / "standards_base.json")
    return s


@pytest.fixture
def extra1():
    d = {
        "user": "pytest",
        "timestamp": "2024-12-18 05:52:01 UTC",
        "platform": "NeXTSTEP-3.3",
        "system": "Windows 3.11 For Workgroups",
        "node": "toaster",
        "python": "3.12.3",
    }
    return d


@pytest.fixture
def extra2():
    d = {
        "user": "pytest again",
        "timestamp": "2024-12-25 05:52:01 UTC",
        "platform": "RaspberryPi 4",
        "system": "System 7",
        "node": "streamer",
        "python": "3.12.4",
    }
    return d


def test_reload(standard):
    assert isinstance(standard.data, dict)
    assert isinstance(standard.metadata, dict)
    assert isinstance(standard.metadata["run_info"], dict)


def test_append_run_info(standard, extra1, extra2):
    assert isinstance(standard.metadata["run_info"], dict)

    standard.append_run_info(extra1)
    assert isinstance(standard.metadata["run_info"], list)

    standard.append_run_info(extra2)
    assert isinstance(standard.metadata["run_info"], list)
    assert len(standard.metadata["run_info"]) == 3


def test_save(standard, extra1, extra2, tmp_path):
    assert isinstance(standard.metadata["run_info"], dict)
    file_path = tmp_path / "new_standard.json"

    # make sure saving doesn't change the contents
    new_standard = Standards(file_path, auto_load=False)
    new_standard.data = standard.data
    new_standard.metadata = standard.metadata
    new_standard.save()
    assert pathlib.Path.exists(file_path)

    newer_standard = Standards(file_path)
    assert newer_standard.data == standard.data
    assert newer_standard.metadata == standard.metadata

    newer_standard.append_run_info(extra1)
    newer_standard.append_run_info(extra2)
    newer_standard.save()
    newer_standard.reload()

    assert isinstance(newer_standard.metadata["run_info"], list)
    assert len(newer_standard.metadata["run_info"]) == 3
    assert newer_standard.metadata["run_info"][0] == standard.metadata["run_info"]
    assert newer_standard.metadata["run_info"][1] == extra1
    assert newer_standard.metadata["run_info"][2] == extra2
