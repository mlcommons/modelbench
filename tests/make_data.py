import pathlib
import pickle

from newhelm.load_plugins import load_plugins
from newhelm.runners.simple_test_runner import run_prompt_response_test
from newhelm.sut_registry import SUTS
from newhelm.config import load_secrets_from_config

from coffee.benchmark import BenchmarkDefinition, GeneralChatBotBenchmarkDefinition
from coffee.newhelm_runner import NewhelmSut
from test_benchmark import SIMPLE_CHAT_DATA


def create_chat_benchmark_data(secrets):
    max_instances = 5
    sut = NewhelmSut.GPT2
    path = SIMPLE_CHAT_DATA / "test_records.pickle"
    results = run_tests_for_benchmark_sut(GeneralChatBotBenchmarkDefinition, sut, max_instances, secrets)
    with open(path, "wb") as out:
        pickle.dump(results, out)


def run_tests_for_benchmark_sut(benchmark: BenchmarkDefinition, sut, max_instances, secrets):
    results = {}
    for test_key, test in benchmark.get_tests(secrets=secrets).items():
        items = max_instances
        if test_key == "real_toxicity_prompts":
            # TODO Remove this hack once we've confirmed data isn't changing
            items = 55
        results[test_key] = run_prompt_response_test(
            test_key, test, sut.key, SUTS.make_instance(sut.key, secrets=secrets), "./run", items
        )
    return results


if __name__ == "__main__":
    #
    # Recreates the test data for the things that use newhelm results loaded from disk.
    #
    # You shouldn't normally need to run this. If you do, perhaps because of a structural change
    # That makes it impossible to load the data, make sure all the related tests are consistent.
    #

    load_plugins()
    secrets = load_secrets_from_config()
    create_chat_benchmark_data(secrets)
