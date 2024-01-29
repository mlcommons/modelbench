import click
from newhelm.base_test import BasePromptResponseTest
from newhelm.benchmark_registry import BENCHMARKS
from newhelm.command_line import (
    DATA_DIR_OPTION,
    SECRETS_FILE_OPTION,
    SUT_OPTION,
    newhelm_cli,
)
from newhelm.general import get_or_create_json_file, to_json
from newhelm.runners.simple_benchmark_runner import (
    SimpleBenchmarkRunner,
    run_prompt_response_test,
)
from newhelm.sut import PromptResponseSUT
from newhelm.sut_registry import SUTS
from newhelm.test_registry import TESTS


@newhelm_cli.command()
@click.option("--test", help="Which registered TEST to run.", required=True)
@SUT_OPTION
@DATA_DIR_OPTION
@SECRETS_FILE_OPTION
def run_test(test: str, sut: str, data_dir: str, secrets: str):
    test_obj = TESTS.make_instance(test)
    sut_obj = SUTS.make_instance(sut)
    secrets_dict = get_or_create_json_file(secrets)
    # Current this only knows how to do prompt response, so assert that is what we have.
    assert isinstance(sut_obj, PromptResponseSUT)
    assert isinstance(test_obj, BasePromptResponseTest)

    test_journal = run_prompt_response_test(test_obj, sut_obj, data_dir, secrets_dict)
    print(to_json(test_journal, indent=4))


@newhelm_cli.command()
@click.option("--benchmark", help="Which registered BENCHMARK to run.")
@SUT_OPTION
@DATA_DIR_OPTION
@SECRETS_FILE_OPTION
def run_benchmark(benchmark, sut, data_dir, secrets):
    benchmark_obj = BENCHMARKS.make_instance(benchmark)
    sut_obj = SUTS.make_instance(sut)
    secrets_dict = get_or_create_json_file(secrets)
    runner = SimpleBenchmarkRunner(data_dir, secrets_dict)
    benchmark_records = runner.run(benchmark_obj, [sut_obj])
    for record in benchmark_records:
        # make it print pretty
        print(to_json(record, indent=4))
