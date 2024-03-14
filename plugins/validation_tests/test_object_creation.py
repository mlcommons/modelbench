import os
import pytest
from newhelm.base_test import BasePromptResponseTest
from newhelm.dependency_helper import FromSourceDependencyHelper
from newhelm.load_plugins import load_plugins
from newhelm.record_init import get_initialization_record
from newhelm.sut_registry import SUTS
from newhelm.test_registry import TESTS
from tests.fake_secrets import fake_all_secrets
from tests.utilities import expensive_tests

# Ensure all the plugins are available during testing.
load_plugins()
_FAKE_SECRETS = fake_all_secrets()


@pytest.mark.parametrize("test_name", [key for key, _ in TESTS.items()])
def test_all_tests_construct_and_record_init(test_name):
    test = TESTS.make_instance(test_name, secrets=_FAKE_SECRETS)
    # This throws if things are set up incorrectly.
    get_initialization_record(test)


# Some tests require such large downloads / complex processing
# that we don't want to do that even on expensive_tests.
# If your Test is timing out, consider adding it here.
TOO_SLOW = {
    "real_toxicity_prompts",
    "bbq",
}


@expensive_tests
@pytest.mark.timeout(10)
@pytest.mark.parametrize(
    "test_name", [key for key, _ in TESTS.items() if key not in TOO_SLOW]
)
def test_all_tests_make_test_items(test_name, tmpdir):
    test = TESTS.make_instance(test_name, secrets=_FAKE_SECRETS)
    if isinstance(test, BasePromptResponseTest):
        test_data_path = os.path.join(tmpdir, test.get_metadata().name)
        dependency_helper = FromSourceDependencyHelper(
            test_data_path,
            test.get_dependencies(),
            required_versions={},
        )

        test_items = test.make_test_items(dependency_helper)
        assert len(test_items) > 0


@pytest.mark.parametrize("sut_name", [key for key, _ in SUTS.items()])
def test_all_suts_construct_and_record_init(sut_name):
    sut = SUTS.make_instance(sut_name, secrets=_FAKE_SECRETS)
    # This throws if things are set up incorrectly.
    get_initialization_record(sut)
