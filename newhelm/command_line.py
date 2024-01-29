import click


@click.group()
def newhelm_cli():
    """To add a command, decorate your function with @newhelm_cli.command()."""
    pass


# Define some reusable options
DATA_DIR_OPTION = click.option(
    "--data-dir",
    default="run_data",
    help="Where to store the auxiliary data produced during the run.",
)

SECRETS_FILE_OPTION = click.option(
    "--secrets", default="secrets/default.json", help="File containing needed secrets."
)

SUT_OPTION = click.option("--sut", help="Which registered SUT to run.", required=True)
