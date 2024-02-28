import os
import shutil
from typing import Optional
import tomli
from importlib import resources
from newhelm import config_templates
from newhelm.secrets_registry import SECRETS, SecretsRegistry


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


def load_secrets_from_config(
    path: str = SECRETS_PATH, registry: SecretsRegistry = SECRETS
) -> None:
    """If the secrets file exists, use it to call SECRETS.set_values."""
    if not os.path.exists(path):
        # Nothing to load
        return
    with open(path, "rb") as f:
        values = tomli.load(f)
    registry.set_values(values)
