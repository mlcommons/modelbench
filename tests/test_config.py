import os
from newhelm.config import (
    DEFAULT_SECRETS,
    load_secrets_from_config,
    write_default_config,
)
from newhelm.secrets_registry import SecretsRegistry


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
    registry = SecretsRegistry()
    registry.register("demo", "demo_api_key", "for the test")

    load_secrets_from_config(secrets_file, registry)

    assert registry.get_required("demo", "demo_api_key") == "12345"


def test_load_secrets_from_config_no_file(tmpdir):
    config_dir = tmpdir.join("config")
    secrets_file = config_dir.join(DEFAULT_SECRETS)
    registry = SecretsRegistry()
    registry.register("demo", "demo_api_key", "for the test")

    load_secrets_from_config(secrets_file, registry)

    # Checks that no values were actually loaded.
    assert registry._values is None
