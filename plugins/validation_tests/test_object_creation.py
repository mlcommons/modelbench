from newhelm.load_plugins import load_plugins
from newhelm.record_init import get_initialization_record
from newhelm.sut_registry import SUTS
from newhelm.test_registry import TESTS

# Ensure all the plugins are available during testing.
load_plugins()


def test_all_tests_construct_and_record_init():
    for _, entry in TESTS.items():
        test = entry.make_instance()
        # This throws if things are set up incorrectly.
        get_initialization_record(test)


def test_all_suts_construct_and_record_init():
    for _, entry in SUTS.items():
        sut = entry.make_instance()
        # This throws if things are set up incorrectly.
        get_initialization_record(sut)
