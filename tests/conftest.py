from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture()
def cwd_tmpdir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def start_time():
    return datetime.now(timezone.utc)


@pytest.fixture
def end_time():
    return datetime.now(timezone.utc) + timedelta(minutes=2)
