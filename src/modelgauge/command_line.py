import pathlib
import pkgutil
import sys

import click
from modellogger.log_config import configure_logging

from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.cli_lazy import LOAD_ALL, LazyModuleImportGroup
from modelgauge.config import write_default_config
from modelgauge.preflight import listify
from modelgauge.sut_factory import SUT_FACTORY
from modelgauge.test_registry import TESTS


@click.group(
    cls=LazyModuleImportGroup,
    name="modelgauge",
    lazy_lists={
        "list": LOAD_ALL,
        "list-annotators": "annotators",
        "list-secrets": LOAD_ALL,
        "list-suts": "suts",
        "list-tests": "tests",
        "run-annotator": "annotators",
    },
)
def cli():
    """Run the ModelGauge library from the command line."""
    # To add a command, decorate your function with @cli.command().
    configure_logging(app_name="modelgauge")
    # Always create the config directory if it doesn't already exist.
    write_default_config()


def display_header(text):
    """Echo the text, but in bold!"""
    click.echo(click.style(text, bold=True))


def display_list_item(text):
    click.echo(f"\t{text}")


def load_local_plugins(_, __, path: pathlib.Path):
    path_str = str(path)
    sys.path.append(path_str)
    plugins = pkgutil.walk_packages([path_str])
    for plugin in plugins:
        __import__(plugin.name)


# Define some reusable options
DATA_DIR_OPTION = click.option(
    "--data-dir",
    default="run_data",
    help="Where to store the auxiliary data produced during the run.",
)

MAX_TEST_ITEMS_OPTION = click.option(
    "-m",
    "--max-test-items",
    default=None,
    type=click.IntRange(1),  # Must be a postive integer
    help="Maximum number of TestItems a Test should run.",
)

LOCAL_PLUGIN_DIR_OPTION = click.option(
    "--plugin-dir",
    type=click.Path(exists=True, dir_okay=True, path_type=pathlib.Path, file_okay=False),
    help="Directory containing plugins to load",
    callback=load_local_plugins,
    expose_value=False,
)

MAX_TOKENS_OPTION = click.option(
    "--max-tokens", default=100, type=click.IntRange(1), help="How many tokens to generate for each completion."
)
TEMP_OPTION = click.option("--temp", default=None, type=float, help="SUT temperature value.")
TOP_P_OPTION = click.option("--top-p", default=None, type=float, help="SUT top-p value.")
TOP_K_OPTION = click.option("--top-k", default=None, type=int, help="SUT top-k value.")


def sut_options_options(func):
    """Adds various SUT options to a command."""
    func = MAX_TOKENS_OPTION(func)
    func = TEMP_OPTION(func)
    func = TOP_P_OPTION(func)
    func = TOP_K_OPTION(func)
    return func


def validate_uid(ctx, param, value):
    """Callback function for click.option UID validation.
    Raises a BadParameter exception if the user-supplied arg(s) are not valid UIDs.
    Applicable for parameters '--test', '--sut', and '--annotator'.
    If no UID is provided (e.g. an empty list or `None`), the value is returned as-is.
    """
    if not value:
        return value
    # Identify what object we are validating UIDs for.
    if "--sut" in param.opts:
        registry = SUT_FACTORY
    elif "--test" in param.opts:
        registry = TESTS
    elif "--annotator" in param.opts:
        registry = ANNOTATORS
    else:
        raise ValueError(f"Cannot validate UID for unknown parameter: {param.opts}")

    # This function handles multi-values and single values.
    values = listify(value)

    unknown_uids = []
    for uid in values:
        if not registry.knows(uid):
            unknown_uids.append(uid)
    if len(unknown_uids) == 0:
        return value

    plurality = "s" if len(unknown_uids) > 1 else ""
    _bad_uid_error(registry, f"Unknown uid{plurality}: '{unknown_uids}'", hint=param.opts)


# this is used for all types of UIDs, not just SUTs
def _bad_uid_error(registry, message, hint=""):
    raise click.BadParameter(
        f"{message}.\nValid options are:\n{registry.compact_uid_list()}",
        param_hint=hint,
    )
