import logging
import os
import pathlib
import warnings
from typing import Optional, Sequence, Type

import click

from modelgauge.annotator import Annotator
from modelgauge.annotator_registry import ANNOTATORS
import modelgauge.annotators.cheval.registration  # noqa: F401
from modelgauge.base_test import PromptResponseTest
from modelgauge.command_line import (  # usort:skip
    DATA_DIR_OPTION,
    LOCAL_PLUGIN_DIR_OPTION,
    MAX_TEST_ITEMS_OPTION,
    display_header,
    display_list_item,
    cli,
    sut_options_options,
)
from modelgauge.config import load_secrets_from_config, toml_format_secrets
from modelgauge.dependency_injection import list_dependency_usage
from modelgauge.general import normalize_filename
from modelgauge.instance_factory import FactoryEntry
from modelgauge.load_namespaces import list_objects
from modelgauge.model_options import ModelOptions
from modelgauge.pipeline_runner import build_runner
from modelgauge.preflight import check_secrets, make_sut
from modelgauge.prompt import TextPrompt
from modelgauge.secret_values import get_all_secrets, RawSecrets
from modelgauge.simple_test_runner import run_prompt_response_test
from modelgauge.single_turn_prompt_response import SUTResponse, TestItem
from modelgauge.sut import PromptResponseSUT
from modelgauge.sut_capabilities_verification import assert_sut_capabilities
from modelgauge.sut_capabilities import AcceptsTextPrompt, ProducesPerTokenLogProbabilities, SUTCapability
from modelgauge.sut_registry import SUTS
from modelgauge.test_registry import TESTS

logger = logging.getLogger(__name__)


@cli.command(name="list")
@LOCAL_PLUGIN_DIR_OPTION
def list_command() -> None:
    """Overview of Modules, Annotators, Tests, and SUTs."""
    loaded_modules = list_objects()
    display_header(f"Loaded Modules: {len(loaded_modules)}")
    for module_name in loaded_modules:
        display_list_item(module_name)
    suts = SUTS.items()
    display_header(f"SUTS: {len(suts)}")
    for sut, sut_entry in suts:
        display_list_item(f"{sut} {sut_entry}")
    annotators = ANNOTATORS.items()
    display_header(f"Annotators: {len(annotators)}")
    for annotator, annotator_entry in annotators:
        display_list_item(f"{annotator} {annotator_entry}")
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


@cli.command()
@LOCAL_PLUGIN_DIR_OPTION
def list_tests() -> None:
    """List details about all registered tests."""
    secrets = load_secrets_from_config()
    for test_uid, test_entry in TESTS.items():
        _display_factory_entry(test_uid, test_entry, secrets)


@cli.command()
@LOCAL_PLUGIN_DIR_OPTION
def list_suts():
    """List details about all registered SUTs (System Under Test)."""
    secrets = load_secrets_from_config()
    for sut_uid, sut in SUTS.items():
        _display_factory_entry(sut_uid, sut, secrets)


@cli.command()
@LOCAL_PLUGIN_DIR_OPTION
def list_annotators():
    """List details about all registered SUTs (System Under Test)."""
    secrets = load_secrets_from_config()
    for annotator_uid, annotator in ANNOTATORS.items():
        _display_factory_entry(annotator_uid, annotator, secrets)


@cli.command()
@LOCAL_PLUGIN_DIR_OPTION
def list_secrets() -> None:
    """List details about secrets modelgauge might need."""
    descriptions = get_all_secrets()
    if descriptions:
        display_header("Here are the known secrets modelgauge might use.")
        click.echo(toml_format_secrets(descriptions))
    else:
        display_header("No secrets used by any installed plugin.")


@cli.command()
@LOCAL_PLUGIN_DIR_OPTION
@click.option("--sut", "-s", help="Which SUT to run.", required=True)
@sut_options_options
@click.option("--prompt", help="The full text to send to the SUT.", required=True)
def run_sut(
    sut: str,
    prompt: str,
    max_tokens: int,
    temp: Optional[float],
    top_logprobs: Optional[int],
    top_p: Optional[float],
    top_k: Optional[int],
):
    """Send a prompt from the command line to a SUT."""
    # TODO Consider a SUT factory that takes in a SUTDefinition and returns a SUT
    options = ModelOptions.create_from_arguments(max_tokens, temp, top_p, top_k, top_logprobs)

    # Current this only knows how to do prompt response, so assert that is what we have.
    sut_instance = make_sut(sut)
    assert isinstance(sut_instance, PromptResponseSUT)
    required_capabilities: Sequence[Type[SUTCapability]] = (
        [AcceptsTextPrompt] if top_logprobs is None else [AcceptsTextPrompt, ProducesPerTokenLogProbabilities]
    )
    assert_sut_capabilities(sut_instance, required_capabilities)

    print(options)
    prompt_instance = TextPrompt(text=prompt)
    request = sut_instance.translate_text_prompt(prompt_instance, options)
    click.echo(f"Native request: {request}\n")
    response = sut_instance.evaluate(request)
    click.echo(f"Native response: {response}\n")
    result = sut_instance.translate_response(request, response)
    click.echo(f"Normalized response: {result.model_dump_json(indent=2)}\n")


@cli.command()
@LOCAL_PLUGIN_DIR_OPTION
@click.option("--annotator", "-a", help="Which annotator to run.", required=True)
@click.option(
    "--prompt",
    help="The prompt that was sent to the SUT.",
    required=False,
    default="Hello",
)
@click.option("--sut-response", help="The SUT's response.", required=False, default="Goodbye")
def run_annotator(
    annotator: str,
    prompt: str,
    sut_response: str,
):
    """Send a prompt and SUT response from the command line to an annotator."""
    secrets = load_secrets_from_config()
    check_secrets(secrets, annotator_uids=[annotator])
    annotator_instance = ANNOTATORS.make_instance(annotator, secrets=secrets)
    assert isinstance(annotator_instance, Annotator)

    test_item = TestItem(prompt=TextPrompt(text=prompt), source_id="cli")
    response = SUTResponse(text=sut_response)

    request = annotator_instance.translate_request(test_item, response)
    click.echo(f"Native request: {request}\n")
    response = annotator_instance.annotate(request)
    click.echo(f"Native response: {response}\n")
    result = annotator_instance.translate_response(request, response)
    click.echo(f"Normalized response: {result.model_dump_json(indent=2)}\n")


@cli.command()
@click.option("--test", "-t", help="Which registered TEST to run.", required=True)
@LOCAL_PLUGIN_DIR_OPTION
@click.option("--sut", "-s", help="Which SUT to run.", required=True)
@DATA_DIR_OPTION
@MAX_TEST_ITEMS_OPTION
@click.option(
    "--output-file",
    help="If specified, will override the default location for outputting the TestRecord.",
)
@click.option(
    "--no-caching",
    is_flag=True,
    show_default=True,
    default=False,
    help="Disable caching.",
)
@click.option(
    "--no-progress-bar",
    is_flag=True,
    show_default=True,
    default=False,
    help="Disable displaying the 'Processing TestItems' progress bar.",
)
def run_test(
    test: str,
    sut: str,
    data_dir: str,
    max_test_items: int,
    output_file: Optional[str],
    no_caching: bool,
    no_progress_bar: bool,
):
    """Run the Test on the desired SUT and output the TestRecord."""
    # Check for missing secrets without instantiating any objects
    secrets = load_secrets_from_config()
    check_secrets(secrets, sut_uids=[sut], test_uids=[test])

    test_obj = TESTS.make_instance(test, secrets=secrets)
    sut_instance = make_sut(sut)

    # Current this only knows how to do prompt response, so assert that is what we have.
    assert isinstance(sut_instance, PromptResponseSUT)
    assert isinstance(test_obj, PromptResponseTest)

    annotator_instances = []
    for annotator in test_obj.get_annotators():
        obj = ANNOTATORS.make_instance(annotator, secrets=secrets)
        assert isinstance(obj, Annotator)
        annotator_instances.append(obj)

    if output_file is None:
        os.makedirs("output", exist_ok=True)
        output_file = os.path.join("output", normalize_filename(f"record_for_{test}_{sut}.json"))
    test_record = run_prompt_response_test(
        test_obj,
        sut_instance,
        annotator_instances,
        data_dir,
        max_test_items,
        use_caching=not no_caching,
        disable_progress_bar=no_progress_bar,
    )
    with open(output_file, "w") as f:
        print(test_record.model_dump_json(indent=4), file=f)
    # For displaying to the screen, clear out the verbose test_item_records
    test_record.test_item_records = []
    print(test_record.model_dump_json(indent=4))
    print("Full TestRecord json written to", output_file)


@cli.command()
@sut_options_options
@click.option(
    "-s",
    "--sut",
    help="Which SUT to run.",
    required=False,
)
@click.option(
    "annotator_uids",
    "-a",
    "--annotator",
    help="Which registered annotator(s) to run",
    multiple=True,
    required=False,
)
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Number of worker threads, default is 10 * number of SUTs.",
)
@click.option(
    "--output-dir",
    "-o",
    default="airr_data/runs",
    type=click.Path(file_okay=False, dir_okay=True, path_type=pathlib.Path),
)
@click.option("--tag", type=str, help="Tag to include in the output directory name.")
@click.option("--debug", is_flag=True, help="Show internal pipeline debugging information.")
@click.option("--jailbreak", is_flag=True, help="Send seed prompts to annotators instead of regular prompts.")
@click.option("--prompt_uid_col", type=str, default=None, help="Column name for prompt UID in the input file.")
@click.option("--prompt_text_col", type=str, default=None, help="Column name for prompt text in the input file.")
@click.option(
    "--seed_prompt_text_col",
    type=str,
    default=None,
    help="Column name for seed prompt text in the input file if using jailbreak mode.",
)
@click.option("--sut_uid_col", type=str, default=None, help="Column name for SUT UID in the input file.")
@click.option("--sut_response_col", type=str, default=None, help="Column name for sut response in the input file.")
@click.argument(
    "input_path",
    type=click.Path(exists=True, path_type=pathlib.Path),
)
def run_job(
    sut,
    annotator_uids,
    workers,
    output_dir,
    tag,
    debug,
    jailbreak,
    input_path,
    max_tokens,
    temp,
    top_p,
    top_k,
    top_logprobs,
    prompt_uid_col,
    prompt_text_col,
    seed_prompt_text_col,
    sut_uid_col,
    sut_response_col,
):
    """Run rows in a CSV through (a) SUT(s) and/or a set of annotators.

    If running a SUT, the file must have 'UID' and 'Text' columns. The output will be saved to a CSV file.
    If running ONLY annotators, the file must have 'UID', 'Prompt', 'SUT', and 'Response' columns. The output will be saved to a json lines file.
    """
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)

    # TODO: break this function up. It's branching too much
    # make sure the job has everything it needs to run
    secrets = load_secrets_from_config()
    if sut:
        sut_options = ModelOptions.create_from_arguments(max_tokens, temp, top_p, top_k, top_logprobs)
        sut_instance = make_sut(sut)
        check_secrets(secrets, annotator_uids=annotator_uids)
    else:
        sut_instance = None
        if max_tokens is not None or temp is not None or top_p is not None or top_k is not None:
            warnings.warn(f"Received SUT options but only running annotators. Options will not be used.")
        check_secrets(secrets, annotator_uids=annotator_uids)
        sut_options = None

    if len(annotator_uids):
        annotators = {
            annotator_uid: ANNOTATORS.make_instance(annotator_uid, secrets=secrets) for annotator_uid in annotator_uids
        }
    else:
        annotators = None

    pipeline_runner = build_runner(
        suts={sut: sut_instance} if sut else None,
        annotators=annotators,
        num_workers=workers,
        input_path=input_path,
        output_dir=output_dir,
        sut_options=sut_options,
        tag=tag,
        prompt_uid_col=prompt_uid_col,
        prompt_text_col=prompt_text_col,
        seed_prompt_text_col=seed_prompt_text_col,
        sut_uid_col=sut_uid_col,
        sut_response_col=sut_response_col,
        jailbreak=jailbreak,
    )

    with click.progressbar(
        length=pipeline_runner.num_total_items,
        label=f"Processing {pipeline_runner.num_input_items} input items"
        + (f" * 1 SUT" if sut else "")
        + (f" * {len(annotators)} annotators" if annotators else "")
        + ":",
    ) as bar:
        last_complete_count = 0

        def show_progress(data):
            nonlocal last_complete_count
            complete_count = data["completed"]
            bar.update(complete_count - last_complete_count)
            last_complete_count = complete_count
            if last_complete_count == pipeline_runner.num_total_items:
                print()  # Print new line after progress bar for better formatting.

        pipeline_runner.run(show_progress, debug)


if __name__ == "__main__":
    cli()
