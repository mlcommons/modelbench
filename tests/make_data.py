import pathlib
import pickle

from newhelm.general import get_or_create_json_file
from newhelm.load_plugins import load_plugins
from newhelm.runners.simple_benchmark_runner import run_prompt_response_test
from newhelm.secrets_registry import SECRETS
from newhelm.sut_registry import SUTS

from coffee.benchmark import BiasHarmDefinition, ToxicityHarmDefinition
from coffee.helm_runner import NewhelmSut
from test_benchmark import SIMPLE_BBQ_DATA, SIMPLE_TOXICITY_DATA


def create_bbq_test_data():
    load_plugins()
    import newhelm.tests.bbq

    harm = BiasHarmDefinition()
    max_instances = 55
    sut = NewhelmSut.GPT2
    counter = 0
    results = {}
    for test in harm.tests():
        items = max_instances
        if isinstance(test, newhelm.tests.bbq.BBQ):
            # BBQ is currently multiple sub-tests, so roughly split the items among them
            items = int(items / len(newhelm.tests.bbq._CATEGORIES))
        results[test] = run_prompt_response_test(
            f"test-{counter}", test, sut.key, SUTS.make_instance(sut.key), "./run", items
        )
        counter += 1
        # serializable = { k.model_dump_json() : results[k].model_dump_json() for k in results }
        with open(SIMPLE_BBQ_DATA / "test_records.pickle", "wb") as out:
            pickle.dump(results, out)


def create_toxicity_test_data():
    load_plugins()
    SECRETS.set_values(get_or_create_json_file(pathlib.Path(__file__).parent.parent / 'secrets/default.json'))

    harm = ToxicityHarmDefinition()
    max_instances = 55
    sut = NewhelmSut.GPT2
    counter = 0
    results = {}
    for test in harm.tests():
        items = max_instances
        results[test] = run_prompt_response_test(
            f"test-{counter}", test, sut.key, SUTS.make_instance(sut.key), "./run", items
        )
        counter += 1
        with open(SIMPLE_TOXICITY_DATA / "test_records.pickle", "wb") as out:
            pickle.dump(results, out)


if __name__ == "__main__":
    # create_bbq_test_data()
    create_toxicity_test_data()
