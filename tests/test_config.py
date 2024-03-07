import os

import pytest
from newhelm.config import (
    DEFAULT_SECRETS,
    load_secrets_from_config,
    write_default_config,
)


def test_write_default_config_writes_files(tmpdir):
    config_dir = tmpdir.join("config")
    write_default_config(config_dir)
    files = [f.basename for f in config_dir.listdir()]
    assert files == ["secrets.toml"]


def test_write_default_config_skips_existing_dir(tmpdir):
    config_dir = tmpdir.join("config")
    os.makedirs(config_dir)
    write_default_config(config_dir)
    files = [f.basename for f in config_dir.listdir()]
    # No files created
    assert files == []


def test_load_secrets_from_config_loads_default(tmpdir):
    config_dir = tmpdir.join("config")
    write_default_config(config_dir)
    secrets_file = config_dir.join(DEFAULT_SECRETS)

    assert load_secrets_from_config(secrets_file) == {"demo": {"demo_api_key": "12345"}}


def test_load_secrets_from_config_no_file(tmpdir):
    config_dir = tmpdir.join("config")
    secrets_file = config_dir.join(DEFAULT_SECRETS)

    with pytest.raises(FileNotFoundError):
        load_secrets_from_config(secrets_file)


def test_load_secrets_from_config_bad_format(tmpdir):
    config_dir = tmpdir.join("config")
    os.makedirs(config_dir)
    secrets_file = config_dir.join(DEFAULT_SECRETS)
    with open(secrets_file, "w") as f:
        f.write("""not_scoped = "some-value"\n""")
    with pytest.raises(AssertionError) as err_info:
        load_secrets_from_config(secrets_file)
    err_text = str(err_info.value)
    assert err_text == "All keys should be in a [scope]."
