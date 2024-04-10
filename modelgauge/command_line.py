import click
from modelgauge.config import write_default_config


@click.group()
def modelgauge_cli():
    """Run the ModelGauge library from the command line."""
    # To add a command, decorate your function with @modelgauge_cli.command().
    # Always create the config directory if it doesn't already exist.
    write_default_config()


def display_header(text):
    """Echo the text, but in bold!"""
    click.echo(click.style(text, bold=True))


def display_list_item(text):
    click.echo(f"\t{text}")


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
