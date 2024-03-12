from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def cwd_tmpdir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Path("config")
    p.mkdir(parents=True, exist_ok=True)
    with (p / "secrets.toml").open("w") as opened_file:
        opened_file.write(
            """
        [perspective_api]
        api_key = ""
        """
        )
    return tmp_path


@pytest.fixture
def start_time():
    return datetime.now(timezone.utc)


@pytest.fixture
def end_time():
    return datetime.now(timezone.utc) + timedelta(minutes=2)
