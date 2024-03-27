import pickle

from newhelm.config import load_secrets_from_config
from newhelm.load_plugins import load_plugins
from newhelm.runners.simple_test_runner import run_prompt_response_test
from newhelm.sut_registry import SUTS

from coffee.benchmark import SafeCaeHazard, SafeCbrHazard
from coffee.newhelm_runner import NewhelmSut
from test_benchmark import SIMPLE_CAE_DATA, SIMPLE_CBR_DATA


def create_cae_test_data(secrets):
    hazard = SafeCaeHazard()
    max_instances = 5
    sut = NewhelmSut.GPT2
    path = SIMPLE_CAE_DATA / "test_records.pickle"
    results = run_tests_for_hazard_on_sut(hazard, sut, max_instances, secrets)
    with open(path, "wb") as out:
        pickle.dump(results, out)


def create_cbr_test_data(secrets):
    hazard = SafeCbrHazard()
    max_instances = 55
    sut = NewhelmSut.GPT2
    results = run_tests_for_hazard_on_sut(hazard, sut, max_instances, secrets)

    with open(SIMPLE_CBR_DATA / "test_records.pickle", "wb") as out:
        pickle.dump(results, out)


def run_tests_for_hazard_on_sut(hazard, sut, max_instances, secrets):
    results = {}
    for test in hazard.tests(secrets):
        items = max_instances
        results[test.uid] = run_prompt_response_test(test, SUTS.make_instance(sut.key, secrets=secrets), "./run", items)
    return results


if __name__ == "__main__":
    #
    # Recreates the test data for the things that use newhelm results loaded from disk.
    #
    # You shouldn't normally need to run this. If you do, perhaps because of a structural change
    # That makes it impossible to load the data, make sure all the related tests are consistent.
    #

    load_plugins()
    secrets = load_secrets_from_config(path="../config/secrets.toml")
    create_cae_test_data(secrets)
    create_cbr_test_data(secrets)
