import pytest
from newhelm.load_plugins import load_plugins
from newhelm.record_init import get_initialization_record
from newhelm.sut_registry import SUTS
from newhelm.test_registry import TESTS
from tests.fake_secrets import fake_all_secrets

# Ensure all the plugins are available during testing.
load_plugins()
_FAKE_SECRETS = fake_all_secrets()


@pytest.mark.parametrize("test_name", [key for key, _ in TESTS.items()])
def test_all_tests_construct_and_record_init(test_name):
    test = TESTS.make_instance(test_name, secrets=_FAKE_SECRETS)
    # This throws if things are set up incorrectly.
    get_initialization_record(test)


@pytest.mark.parametrize("sut_name", [key for key, _ in SUTS.items()])
def test_all_suts_construct_and_record_init(sut_name):
    sut = SUTS.make_instance(sut_name, secrets=_FAKE_SECRETS)
    # This throws if things are set up incorrectly.
    get_initialization_record(sut)
