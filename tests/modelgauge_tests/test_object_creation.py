import os

import pytest
from flaky import flaky  # type: ignore

from modelgauge.base_test import PromptResponseTest
from modelgauge.config import load_secrets_from_config
from modelgauge.dependency_helper import FromSourceDependencyHelper
from modelgauge.load_namespaces import load_namespaces
from modelgauge.record_init import InitializationRecord
from modelgauge.sut_registry import SUTS
from modelgauge.test_registry import TESTS
from modelgauge_tests.fake_secrets import fake_all_secrets
from modelgauge_tests.utilities import expensive_tests

# Ensure all the modules are available during testing.
load_namespaces()

_FAKE_SECRETS = fake_all_secrets()
TIMEOUT = 25 * 60  # 25 minutes. Baseten is the slowest to start up.


@pytest.fixture(scope="session")
def shared_run_dir(tmp_path_factory):
    # Create a single tmpdir and have all `make_test_items` share it.
    return tmp_path_factory.mktemp("run_data")


# Some tests require such large downloads / complex processing
# that we don't want to do that even on expensive_tests.
# If your Test is timing out, consider adding it here.
TOO_SLOW = {}


@expensive_tests
@pytest.mark.timeout(30)
@flaky
@pytest.mark.parametrize("test_name", [key for key, _ in TESTS.items() if key not in TOO_SLOW])
def test_all_tests_make_test_items(test_name, shared_run_dir):
    test = TESTS.make_instance(test_name, secrets=load_secrets_from_config(path="."))

    if isinstance(test, PromptResponseTest):
        test_data_path = os.path.join(shared_run_dir, test.__class__.__name__)
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
    assert hasattr(sut, "initialization_record"), "SUT is probably missing @modelgauge_sut() decorator."
    assert isinstance(sut.initialization_record, InitializationRecord)
