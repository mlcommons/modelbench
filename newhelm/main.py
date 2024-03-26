from typing import Optional
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
from newhelm.instance_factory import FactoryEntry

from newhelm.load_plugins import load_plugins, list_plugins
from newhelm.prompt import SUTOptions, TextPrompt
from newhelm.secret_values import MissingSecretValues, RawSecrets, get_all_secrets
from newhelm.sut import PromptResponseSUT
from newhelm.dependency_injection import list_dependency_usage
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


def _display_factory_entry(uid: str, entry: FactoryEntry, secrets: RawSecrets):
    def format_missing_secrets(missing):
        """Return formatted string for missing secrets."""
        return "\n".join(
            f"Scope: '{secret['scope']}', Key: '{secret['key']}', Instructions: '{secret['instructions']}'"
            for secret in missing
        )

    used, missing = list_dependency_usage(entry.args, entry.kwargs, secrets)
    missing = format_missing_secrets(missing)

    display_header(uid)
    click.echo(f"Class: {entry.cls.__name__}")
    click.echo(f"Args: {entry.args}")
    click.echo(f"Kwargs: {entry.kwargs}")

    if used:
        click.echo("Used Secrets:")
        click.echo(used)
    else:
        click.echo("No Secrets Used.")

    if missing:
        click.echo("Missing Secrets:")
        click.echo(missing)

    click.echo()


@newhelm_cli.command()
def list_tests() -> None:
    """List details about all registered tests."""
    secrets = load_secrets_from_config()
    for test_uid, test_entry in TESTS.items():
        _display_factory_entry(test_uid, test_entry, secrets)


@newhelm_cli.command()
def list_suts():
    """List details about all registered SUTs (System Under Test)."""
    secrets = load_secrets_from_config()
    for sut_uid, sut in SUTS.items():
        _display_factory_entry(sut_uid, sut, secrets)


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
@click.option(
    "--num-completions",
    default=None,
    type=click.IntRange(1),
    help="How many different completions to generation.",
)
@click.option(
    "--max-tokens",
    default=None,
    type=click.IntRange(1),
    help="How many tokens to generate for each completion.",
)
@click.option(
    "--top-logprobs",
    type=click.IntRange(1),
    help="How many log probabilities to report for each token position.",
)
def run_sut(
    sut: str,
    prompt: str,
    num_completions: Optional[int],
    max_tokens: Optional[int],
    top_logprobs: Optional[int],
):
    """Send a prompt from the command line to a SUT."""
    secrets = load_secrets_from_config()
    try:
        sut_obj = SUTS.make_instance(sut, secrets=secrets)
    except MissingSecretValues as e:
        raise_if_missing_from_config([e])

    # Current this only knows how to do prompt response, so assert that is what we have.
    assert isinstance(sut_obj, PromptResponseSUT)

    options = SUTOptions()
    if num_completions:
        options.num_completions = num_completions
    if max_tokens:
        options.max_tokens = max_tokens
    if top_logprobs:
        options.top_logprobs = top_logprobs
    prompt_obj = TextPrompt(text=prompt, options=options)
    request = sut_obj.translate_text_prompt(prompt_obj)
    click.echo(f"Native request: {request}\n")
    response = sut_obj.evaluate(request)
    click.echo(f"Native response: {response}\n")
    result = sut_obj.translate_response(request, response)
    click.echo(f"Normalized response: {result.model_dump_json(indent=2)}\n")


if __name__ == "__main__":
    load_plugins()
    newhelm_cli()
