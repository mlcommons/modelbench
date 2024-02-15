import pathlib
import pickle

from newhelm.general import get_or_create_json_file
from newhelm.load_plugins import load_plugins
from newhelm.runners.simple_benchmark_runner import run_prompt_response_test
from newhelm.secrets_registry import SECRETS
from newhelm.sut_registry import SUTS

from coffee.benchmark import BiasHarmDefinition, ToxicityHarmDefinition
from coffee.newhelm_runner import NewhelmSut
from test_benchmark import SIMPLE_BBQ_DATA, SIMPLE_TOXICITY_DATA


def create_bbq_test_data():
    harm = BiasHarmDefinition()
    max_instances = 5
    sut = NewhelmSut.GPT2
    path = SIMPLE_BBQ_DATA / "test_records.pickle"
    results = run_tests_for_harm_on_sut(harm, sut, max_instances)
    with open(path, "wb") as out:
        pickle.dump(results, out)


def create_toxicity_test_data():
    harm = ToxicityHarmDefinition()
    max_instances = 55
    sut = NewhelmSut.GPT2
    results = run_tests_for_harm_on_sut(harm, sut, max_instances)

    with open(SIMPLE_TOXICITY_DATA / "test_records.pickle", "wb") as out:
        pickle.dump(results, out)


def run_tests_for_harm_on_sut(harm, sut, max_instances):
    results = {}
    for counter, test in enumerate(harm.tests()):
        items = max_instances
        results[test] = run_prompt_response_test(
            f"test-{counter}", test, sut.key, SUTS.make_instance(sut.key), "./run", items
        )
        counter += 1
    return results


if __name__ == "__main__":
    #
    # Recreates the test data for the things that use newhelm results loaded from disk.
    #
    # You shouldn't normally need to run this. If you do, perhaps because of a strucutral change
    # That makes it impossible to laod the data, make sure all the related tests are consistent.
    #

    load_plugins()
    SECRETS.set_values(get_or_create_json_file(pathlib.Path(__file__).parent.parent / "secrets/default.json"))

    create_bbq_test_data()
    create_toxicity_test_data()
