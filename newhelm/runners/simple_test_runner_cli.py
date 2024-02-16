import click
from newhelm.base_test import BasePromptResponseTest
from newhelm.command_line import (
    DATA_DIR_OPTION,
    MAX_TEST_ITEMS_OPTION,
    SECRETS_FILE_OPTION,
    SUT_OPTION,
    newhelm_cli,
)
from newhelm.general import get_or_create_json_file
from newhelm.runners.simple_test_runner import (
    run_prompt_response_test,
)
from newhelm.secrets_registry import SECRETS
from newhelm.sut import PromptResponseSUT
from newhelm.sut_registry import SUTS
from newhelm.test_registry import TESTS


@newhelm_cli.command()
@click.option("--test", help="Which registered TEST to run.", required=True)
@SUT_OPTION
@DATA_DIR_OPTION
@SECRETS_FILE_OPTION
@MAX_TEST_ITEMS_OPTION
def run_test(test: str, sut: str, data_dir: str, secrets: str, max_test_items: int):
    test_obj = TESTS.make_instance(test)
    sut_obj = SUTS.make_instance(sut)
    SECRETS.set_values(get_or_create_json_file(secrets))
    # Current this only knows how to do prompt response, so assert that is what we have.
    assert isinstance(sut_obj, PromptResponseSUT)
    assert isinstance(test_obj, BasePromptResponseTest)

    test_journal = run_prompt_response_test(
        test, test_obj, sut, sut_obj, data_dir, max_test_items
    )
    print(test_journal.model_dump_json(indent=4))
