import pytest


@pytest.fixture()
def cwd_tmpdir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    return tmp_path
