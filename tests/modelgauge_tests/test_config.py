import os
import pytest
from modelgauge.config import (
    DEFAULT_SECRETS,
    MissingSecretsFromConfig,
    load_secrets_from_config,
    raise_if_missing_from_config,
    write_default_config,
)
from modelgauge.secret_values import MissingSecretValues, SecretDescription


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

    assert load_secrets_from_config(secrets_file) == {"demo": {"api_key": "12345"}}


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


def test_raise_if_missing_from_config_nothing_on_empty():
    raise_if_missing_from_config([])


def test_raise_if_missing_from_config_single():
    secret = SecretDescription(
        scope="some-scope", key="some-key", instructions="some-instructions"
    )
    missing = MissingSecretValues([secret])
    with pytest.raises(MissingSecretsFromConfig) as err_info:
        raise_if_missing_from_config([missing], config_path="some/path.toml")

    assert (
        str(err_info.value)
        == """\
To perform this run you need to add the following values to your secrets file 'some/path.toml':
[some-scope]
# some-instructions
some-key="<value>"
"""
    )


def test_raise_if_missing_from_config_combines():
    scope1_key1 = SecretDescription(
        scope="scope1", key="key1", instructions="instructions1"
    )
    scope1_key2 = SecretDescription(
        scope="scope1", key="key2", instructions="instructions2"
    )
    scope2_key1 = SecretDescription(
        scope="scope2", key="key1", instructions="instructions3"
    )
    missing = [
        # Out of order
        MissingSecretValues([scope1_key1]),
        MissingSecretValues([scope2_key1]),
        MissingSecretValues([scope1_key2]),
    ]
    with pytest.raises(MissingSecretsFromConfig) as err_info:
        raise_if_missing_from_config(missing, config_path="some/path.toml")

    assert (
        str(err_info.value)
        == """\
To perform this run you need to add the following values to your secrets file 'some/path.toml':
[scope1]
# instructions1
key1="<value>"
# instructions2
key2="<value>"

[scope2]
# instructions3
key1="<value>"
"""
    )
