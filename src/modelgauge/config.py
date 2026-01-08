import os
import pathlib
import shutil
from importlib import resources
from typing import Dict, Mapping, Sequence

import tomli
from modellogger.log_config import get_logger

from modelgauge import config_templates
from modelgauge.secret_values import MissingSecretValues, RawSecrets, SecretDescription

DEFAULT_CONFIG_DIR = "config"
DEFAULT_SECRETS = "secrets.toml"
SECRETS_PATH = os.path.join(DEFAULT_CONFIG_DIR, DEFAULT_SECRETS)
CONFIG_TEMPLATES = [DEFAULT_SECRETS]

logger = get_logger(__name__)


def find_config_dir(path: str = ".") -> str:
    """Search up the tree for the config directory."""
    current_dir = os.path.abspath(path)
    while True:
        config_dir = os.path.join(current_dir, DEFAULT_CONFIG_DIR)
        if os.path.exists(config_dir):
            return config_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Reached root directory
            raise FileNotFoundError(
                f"Could not find the config directory '{DEFAULT_CONFIG_DIR}' anywhere along the path to '{path}'."
            )
        current_dir = parent_dir


def write_default_config(parent_dir: str = "."):
    """If the config directory doesn't exist, fill it with defaults."""
    try:
        find_config_dir(parent_dir)
        # Don't do anything if the config directory already exists.
        # Assume if it exists we don't need to add templates
    except FileNotFoundError:
        dir = os.path.join(parent_dir, DEFAULT_CONFIG_DIR)
        os.makedirs(dir)
        for template in CONFIG_TEMPLATES:
            source_file = str(resources.files(config_templates) / template)
            output_file = os.path.join(dir, template)
            shutil.copyfile(source_file, output_file)


def load_secrets_from_config(path: str = ".") -> RawSecrets:
    """Load the toml file and verify it is shaped as expected."""
    try:
        secrets_path = os.path.join(find_config_dir(path), DEFAULT_SECRETS)
        with open(secrets_path, "rb") as f:
            data = tomli.load(f)
    except FileNotFoundError as exc:
        logger.warning("Could not find secrets file.")
        data = {}
    for values in data.values():
        # Verify the config is shaped as expected.
        assert isinstance(values, Mapping), "All keys should be in a [scope]."
        for key, value in values.items():
            assert isinstance(key, str)
            assert isinstance(value, str)
    return data


def toml_format_secrets(secrets: Sequence[SecretDescription]) -> str:
    """Format the secrets as they'd appear in a toml file.

    All values are set to "<value>".
    """

    scopes: Dict[str, Dict[str, str]] = {}
    for secret in secrets:
        if secret.scope not in scopes:
            scopes[secret.scope] = {}
        scopes[secret.scope][secret.key] = secret.instructions
    scope_displays = []
    for scope, in_scope in sorted(scopes.items()):
        scope_display = f"[{scope}]\n"
        for key, instruction in sorted(in_scope.items()):
            scope_display += f"# {instruction}\n"
            scope_display += f'{key}="<value>"\n'
        scope_displays.append(scope_display)
    return "\n".join(scope_displays)


class MissingSecretsFromConfig(MissingSecretValues):
    """Exception showing how to add missing secrets to the config file."""

    def __init__(self, missing: MissingSecretValues, config_path: str = SECRETS_PATH):
        super().__init__(descriptions=missing.descriptions)
        self.config_path = config_path

    def __str__(self):
        message = f"To perform this run you need to add the following values "
        message += f"to your secrets file '{self.config_path}':\n"
        message += toml_format_secrets(self.descriptions)
        return message


def raise_if_missing_from_config(missing_values: Sequence[MissingSecretValues], config_path: str = SECRETS_PATH):
    """If there are missing secrets, raise a MissingSecretsFromConfig exception."""
    if not missing_values:
        return
    combined = MissingSecretValues.combine(missing_values)
    raise MissingSecretsFromConfig(combined, str(pathlib.Path(config_path).absolute()))
