import pathlib
import pkgutil
import sys

import click
from modelgauge.config import write_default_config
from modelgauge.load_plugins import load_plugins


@click.group(name="modelgauge")
def modelgauge_cli():
    """Run the ModelGauge library from the command line."""
    # To add a command, decorate your function with @modelgauge_cli.command().

    # Always create the config directory if it doesn't already exist.
    write_default_config()

    # We need to call `load_plugins` before the cli in order to:
    # * Allow plugins to add their own CLI commands
    # * Enable --help to correctly list command options (e.g. valid values for SUT)
    load_plugins()


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

SUT_OPTION = click.option("--sut", help="Which registered SUT to run.", required=True)

LOCAL_PLUGIN_DIR_OPTION = click.option(
    "--plugin-dir",
    type=click.Path(
        exists=True, dir_okay=True, path_type=pathlib.Path, file_okay=False
    ),
    help="Directory containing plugins to load",
    callback=load_local_plugins,
    expose_value=False,
)
