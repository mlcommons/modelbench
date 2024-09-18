from datetime import datetime, timedelta, timezone
from typing import Dict

import pytest
from modelgauge.secret_values import (
    get_all_secrets,
)


@pytest.fixture()
def cwd_tmpdir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture()
def fake_secrets(value="some-value"):
    secrets = get_all_secrets()
    raw_secrets: Dict[str, Dict[str, str]] = {}
    for secret in secrets:
        if secret.scope not in raw_secrets:
            raw_secrets[secret.scope] = {}
        raw_secrets[secret.scope][secret.key] = value
    return raw_secrets


@pytest.fixture
def start_time():
    return datetime.now(timezone.utc)


@pytest.fixture
def end_time():
    return datetime.now(timezone.utc) + timedelta(minutes=2)


def pytest_addoption(parser):
    parser.addoption(
        "--expensive-tests",
        action="store_true",
        dest="expensive-tests",
        help="enable expensive tests",
    )
