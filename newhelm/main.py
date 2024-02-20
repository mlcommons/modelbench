import click

from newhelm.command_line import (
    SECRETS_FILE_OPTION,
    SUT_OPTION,
    display_header,
    display_list_item,
    newhelm_cli,
)
from newhelm.general import get_or_create_json_file

from newhelm.load_plugins import load_plugins, list_plugins
from newhelm.prompt import Prompt
from newhelm.secrets_registry import SECRETS
from newhelm.sut import PromptResponseSUT
from newhelm.sut_registry import SUTS
from newhelm.test_registry import TESTS


@newhelm_cli.command()
def list() -> None:
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


# TODO: Consider moving this somewhere else.
@newhelm_cli.command()
@SUT_OPTION
@SECRETS_FILE_OPTION
@click.option("--prompt", help="The full text to send to the SUT.")
def run_sut(sut: str, secrets: str, prompt: str):
    sut_obj = SUTS.make_instance(sut)
    SECRETS.set_values(get_or_create_json_file(secrets))
    # Current this only knows how to do prompt response, so assert that is what we have.
    assert isinstance(sut_obj, PromptResponseSUT)

    prompt_obj = Prompt(text=prompt)
    request = sut_obj.translate_request(prompt_obj)
    click.echo(f"{request}\n")
    response = sut_obj.evaluate(request)
    click.echo(f"{response}\n")
    result = sut_obj.translate_response(prompt_obj, response)
    click.echo(f"{result}\n")


if __name__ == "__main__":
    load_plugins()
    newhelm_cli()
