import click

from newhelm.command_line import (
    SUT_OPTION,
    display_header,
    display_list_item,
    newhelm_cli,
)
from newhelm.config import (
    load_secrets_from_config,
    raise_if_missing_from_config,
    toml_format_secrets,
)

from newhelm.load_plugins import load_plugins, list_plugins
from newhelm.prompt import TextPrompt
from newhelm.secret_values import MissingSecretValues, get_all_secrets
from newhelm.sut import PromptResponseSUT
from newhelm.sut_registry import SUTS
from newhelm.test_registry import TESTS


@newhelm_cli.command()
def list() -> None:
    """Overview of Plugins, Tests, and SUTs."""
    plugins = list_plugins()
    display_header(f"Plugin Modules: {len(plugins)}")
    for module_name in plugins:
        display_list_item(module_name)
    suts = SUTS.items()
    display_header(f"SUTS: {len(suts)}")
    for sut, sut_entry in suts:
        display_list_item(f"{sut} {sut_entry}")
    tests = TESTS.items()
    display_header(f"Tests: {len(tests)}")
    for test, test_entry in tests:
        display_list_item(f"{test} {test_entry}")


@newhelm_cli.command()
def list_tests() -> None:
    """List details about all registered tests."""
    secrets = load_secrets_from_config()
    for test, test_entry in TESTS.items():
        try:
            test_obj = test_entry.make_instance(secrets=secrets)
        except MissingSecretValues as e:
            display_header(
                f"Cannot display test {test} because it requires the following secrets:"
            )
            for secret in e.descriptions:
                click.echo(secret)
            click.echo()
            continue

        metadata = test_obj.get_metadata()
        display_header(metadata.name)
        click.echo(f"Command line key: {test}")
        click.echo(f"Description: {metadata.description}")
        click.echo()


@newhelm_cli.command()
def list_secrets() -> None:
    """List details about secrets newhelm might need."""
    descriptions = get_all_secrets()
    if descriptions:
        display_header("Here are the known secrets newhelm might use.")
        click.echo(toml_format_secrets(descriptions))
    else:
        display_header("No secrets used by any installed plugin.")


# TODO: Consider moving this somewhere else.
@newhelm_cli.command()
@SUT_OPTION
@click.option("--prompt", help="The full text to send to the SUT.")
def run_sut(sut: str, prompt: str):
    """Send a prompt from the command line to a SUT."""
    secrets = load_secrets_from_config()
    try:
        sut_obj = SUTS.make_instance(sut, secrets=secrets)
    except MissingSecretValues as e:
        raise_if_missing_from_config([e])

    # Current this only knows how to do prompt response, so assert that is what we have.
    assert isinstance(sut_obj, PromptResponseSUT)

    prompt_obj = TextPrompt(text=prompt)
    request = sut_obj.translate_text_prompt(prompt_obj)
    click.echo(f"Native request: {request}\n")
    response = sut_obj.evaluate(request)
    click.echo(f"Native response: {response}\n")
    result = sut_obj.translate_response(request, response)
    click.echo(f"Normalized response: {result}\n")


if __name__ == "__main__":
    load_plugins()
    newhelm_cli()
