import pathlib
import pkgutil
import sys
from typing import List

import click
from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.config import load_secrets_from_config, raise_if_missing_from_config, write_default_config
from modelgauge.dynamic_sut_finder import make_dynamic_sut_for
from modelgauge.load_plugins import load_plugins
from modelgauge.secret_values import MissingSecretValues
from modelgauge.sut import SUTNotFoundException, SUTOptions
from modelgauge.sut_registry import SUTS
from modelgauge.test_registry import TESTS


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


def compact_uid_list(registry) -> str:
    valid_uids = sorted(registry.keys(), key=lambda x: x.lower())
    valid_uids_str = "\n\t".join(valid_uids)
    return "\t" + valid_uids_str


def listify(value):
    """Some functions accept a single UID or a list of them."""
    if isinstance(value, str):
        return [value]
    return value


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
    "--max-tokens", default=None, type=click.IntRange(1), help="How many tokens to generate for each completion."
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


def create_sut_options(max_tokens, temp, top_p, top_k):
    options = SUTOptions()
    if max_tokens is not None:
        options.max_tokens = max_tokens
    if temp is not None:
        options.temperature = temp
    if top_p is not None:
        options.top_p = top_p
    if top_k is not None:
        options.top_k_per_token = top_k
    return options


def _validate_sut_uid(ctx, param, value):
    if "--sut" not in param.opts:
        raise ValueError(f"This function validates SUT UIDs, not {param.opts}")
    # A blank sut uid is OK for some invocations of modelgauge.
    # Commands where a non-blank sut uid is required must enforce that with required=True
    if not value:
        return value

    # This function handles multi-values and single values.
    # modelgauge and modelbench runs now only support one SUT at a time,
    # and that is enforced in the input validation. This function can still
    # handle multiple SUT UIDs at a time to match validate_uid's semantics,
    # and because it doesn't know or care about the logic in the caller.
    sut_uids = listify(value)
    requested_sut_uids = classify_sut_uids(sut_uids)
    valid_sut_uids = requested_sut_uids["known"]
    for sut_uid in requested_sut_uids["dynamic"]:
        dynamic_sut = make_dynamic_sut_for(sut_uid)  # a tuple that can be splatted for SUTS.register
        if dynamic_sut:
            SUTS.register(*dynamic_sut)
            valid_sut_uids.append(dynamic_sut[1])
        else:
            requested_sut_uids["unknown"].append(sut_uid)

    # We don't raise click exceptions, because those aren't caught by pytest.raises in unit tests
    if len(requested_sut_uids["unknown"]) > 0:
        raise SUTNotFoundException(f"Invalid SUT UIDs found in {', '.join(requested_sut_uids['unknown'])}")

    valid_sut_uids = list(set(valid_sut_uids))  # dedupe
    if not valid_sut_uids:
        raise SUTNotFoundException(f'No valid SUT UIDs found in "{value}"')

    if isinstance(value, str):
        return valid_sut_uids[0]
    return valid_sut_uids


def validate_uid(ctx, param, value):
    """Callback function for click.option UID validation.
    Raises a BadParameter exception if the user-supplied arg(s) are not valid UIDs.
    Applicable for parameters '--test', '--sut', and '--annotator'.
    SUT IDs are validated in _validate_sut_uid via this function.
    If no UID is provided (e.g. an empty list or `None`), the value is returned as-is.
    """
    if not value:
        return value
    # Identify what object we are validating UIDs for.
    if "--sut" in param.opts:
        return _validate_sut_uid(ctx, param, value)
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
        if uid not in registry.keys():
            unknown_uids.append(uid)
    if len(unknown_uids) == 0:
        return value

    plurality = "s" if len(unknown_uids) > 1 else ""
    _bad_uid_error(registry, f"Unknown uid{plurality}: '{unknown_uids}'", hint=param.opts)


def get_missing_secrets(secrets, registry, uids):
    missing_secrets: List[MissingSecretValues] = []
    for uid in uids:
        missing_secrets.extend(registry.get_missing_dependencies(uid, secrets=secrets))
    return missing_secrets


def check_secrets(secrets, sut_uids=None, test_uids=None, annotator_uids=None):
    """Checks if all secrets are present for the given UIDs. Raises an error and reports all missing secrets."""
    missing_secrets: List[MissingSecretValues] = []
    if sut_uids is not None:
        missing_secrets.extend(get_missing_secrets(secrets, SUTS, listify(sut_uids)))
    if test_uids is not None:
        missing_secrets.extend(get_missing_secrets(secrets, TESTS, test_uids))
        # Check secrets for the annotators in the test as well.
        for test_uid in test_uids:
            test_cls = TESTS._get_entry(test_uid).cls
            missing_secrets.extend(get_missing_secrets(secrets, ANNOTATORS, test_cls.get_annotators()))
    if annotator_uids is not None:
        missing_secrets.extend(get_missing_secrets(secrets, ANNOTATORS, annotator_uids))
    raise_if_missing_from_config(missing_secrets)
    return True


def classify_sut_uids(uids):
    """The CLI now accepts dynamic SUT ids (e.g. "deepseek-ai/DeepSeek-V3:together:hfrelay") in addition to
    pre-registered SUT ids (e.g. "phi-3.5-moe-instruct"). SUT creation and validation are different
    between those two types. This function returns the SUT ids organized by type."""
    uids = listify(uids)
    if len(uids) < 1:
        _bad_uid_error(SUTS, "Please provide at least one SUT uid.")
    identified = {"known": [], "dynamic": [], "unknown": []}
    for uid in uids:
        if uid in SUTS.keys():
            identified["known"].append(uid)
        elif ":" in uid:
            identified["dynamic"].append(uid)
        else:
            identified["unknown"].append(uid)
    return identified


# TODO: this this really belong in a module named command_line?
def make_sut(sut_uid: str):
    """Checks that user has all required secrets and returns instantiated SUT."""
    secrets = load_secrets_from_config()
    check_secrets(secrets, sut_uids=[sut_uid])
    sut = SUTS.make_instance(sut_uid, secrets=secrets)
    return sut


# this is used for all types of UIDs, not just SUTs
def _bad_uid_error(registry, message, hint=""):
    raise click.BadParameter(
        f"{message}.\nValid options are:\n{compact_uid_list(registry)}",
        param_hint=hint,
    )
