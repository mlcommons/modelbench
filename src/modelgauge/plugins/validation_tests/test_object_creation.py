import os
import pytest
from flaky import flaky  # type: ignore
from modelgauge.base_test import PromptResponseTest
from modelgauge.config import load_secrets_from_config
from modelgauge.dependency_helper import FromSourceDependencyHelper
from modelgauge.load_plugins import load_plugins
from modelgauge.prompt import SUTOptions, TextPrompt
from modelgauge.record_init import InitializationRecord
from modelgauge.sut import PromptResponseSUT, SUTResponse
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.sut_registry import SUTS
from modelgauge.test_registry import TESTS
from tests.fake_secrets import fake_all_secrets
from tests.utilities import expensive_tests

# Ensure all the plugins are available during testing.
load_plugins()
_FAKE_SECRETS = fake_all_secrets()


@pytest.mark.parametrize("test_name", [key for key, _ in TESTS.items()])
def test_all_tests_construct_and_record_init(test_name):
    test = TESTS.make_instance(test_name, secrets=_FAKE_SECRETS)
    assert hasattr(
        test, "initialization_record"
    ), "Test is probably missing @modelgauge_test() decorator."
    assert isinstance(test.initialization_record, InitializationRecord)


@pytest.fixture(scope="session")
def shared_run_dir(tmp_path_factory):
    # Create a single tmpdir and have all `make_test_items` share it.
    return tmp_path_factory.mktemp("run_data")


# Some tests require such large downloads / complex processing
# that we don't want to do that even on expensive_tests.
# If your Test is timing out, consider adding it here.
TOO_SLOW = {
    "real_toxicity_prompts",
    "bbq",
}


@expensive_tests
@pytest.mark.timeout(30)
@flaky
@pytest.mark.parametrize(
    "test_name", [key for key, _ in TESTS.items() if key not in TOO_SLOW]
)
def test_all_tests_make_test_items(test_name, shared_run_dir):
    test = TESTS.make_instance(test_name, secrets=_FAKE_SECRETS)
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
    assert hasattr(
        sut, "initialization_record"
    ), "SUT is probably missing @modelgauge_sut() decorator."
    assert isinstance(sut.initialization_record, InitializationRecord)


# This test can take a while, and we don't want a test run to fail
# just because an external service is being slow. So we set a somewhat
# high timeout value that gives the test a chance to complete most of the time,
# but still fails if the external service really is flaky or slow, so we can
# get a sense of a real user's experience.
@expensive_tests
@pytest.mark.timeout(45)
@pytest.mark.parametrize("sut_name", [key for key, _ in SUTS.items()])
def test_all_suts_can_evaluate(sut_name):
    sut = SUTS.make_instance(sut_name, secrets=load_secrets_from_config())
    assert isinstance(sut, PromptResponseSUT), "Update this test to handle other types."
    if AcceptsTextPrompt in sut.capabilities:
        native_request = sut.translate_text_prompt(
            TextPrompt(
                text="What is your name?",
                options=SUTOptions(max_tokens=3, num_completions=1),
            )
        )
    else:
        raise AssertionError("Update test to handle other kinds of prompts.")
    native_response = sut.evaluate(native_request)
    response = sut.translate_response(native_request, native_response)
    assert isinstance(response, SUTResponse)
    assert response.completions[0].text.strip() != ""
