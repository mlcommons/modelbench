import copy
import importlib
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict
from unittest import mock

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


# This monkeypatch makes it possible to run the tests without having to have an actual config file and should work
# with any additional secrets going forward. It has to be weird because it has to be done before the import of
# sut_factory as secrets are loaded during the import of the module, when the SUT_FACTORY is instantiated.


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    import modelgauge.config as mg_config

    mock_secret = defaultdict(lambda: defaultdict(lambda: "fake-secret"))
    mock_secret["demo"] = {"api_key": "12345"}

    original_func = copy.copy(mg_config.load_secrets_from_config)

    def new_func(path=None):
        if not path:
            return mock_secret
        else:
            return original_func(path)

    mg_config.load_secrets_from_config = new_func
    if "modelgauge.sut_factory" in sys.modules:
        importlib.reload(sys.modules["modelgauge.sut_factory"])


actual_time_sleep = time.sleep


@pytest.fixture(scope="session", autouse=True)
def sleep_faster():
    with mock.patch("time.sleep", lambda x: actual_time_sleep(x / 100000)) as _fixture:
        yield _fixture
