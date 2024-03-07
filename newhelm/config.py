import os
import shutil
from typing import Mapping
import tomli
from importlib import resources
from newhelm import config_templates
from newhelm.secret_values import RawSecrets


DEFAULT_CONFIG_DIR = "config"
DEFAULT_SECRETS = "secrets.toml"
SECRETS_PATH = os.path.join(DEFAULT_CONFIG_DIR, DEFAULT_SECRETS)
CONFIG_TEMPLATES = [DEFAULT_SECRETS]


def write_default_config(dir: str = DEFAULT_CONFIG_DIR):
    """If the config directory doesn't exist, fill it with defaults."""
    if os.path.exists(dir):
        # Assume if it exists we don't need to add templates
        return
    os.makedirs(dir)
    for template in CONFIG_TEMPLATES:
        source_file = str(resources.files(config_templates) / template)
        output_file = os.path.join(dir, template)
        shutil.copyfile(source_file, output_file)


def load_secrets_from_config(path: str = SECRETS_PATH) -> RawSecrets:
    """Load the toml file and verify it is shaped as expected."""
    with open(path, "rb") as f:
        data = tomli.load(f)
    for values in data.values():
        # Verify the config is shaped as expected.
        assert isinstance(values, Mapping), "All keys should be in a [scope]."
        for key, value in values.items():
            assert isinstance(key, str)
            assert isinstance(value, str)
    return data
