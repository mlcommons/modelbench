import os
import re

import pytest
from flaky import flaky  # type: ignore
from modelgauge.base_test import PromptResponseTest
from modelgauge.caching import SqlDictCache
from modelgauge.config import load_secrets_from_config
from modelgauge.dependency_helper import FromSourceDependencyHelper
from modelgauge.external_data import WebData
from modelgauge.load_plugins import load_plugins
from modelgauge.locales import EN_US  # see "workaround" below
from modelgauge.prompt import SUTOptions, TextPrompt
from modelgauge.prompt_sets import demo_prompt_set_url
from modelgauge.record_init import InitializationRecord
from modelgauge.sut import PromptResponseSUT, SUTResponse
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.sut_registry import SUTS

from modelgauge.suts.huggingface_chat_completion import HUGGING_FACE_TIMEOUT
from modelgauge.test_registry import TESTS
from modelgauge.tests.safe_v1 import BaseSafeTestVersion1  # see "workaround" below
from modelgauge_tests.fake_secrets import fake_all_secrets
from modelgauge_tests.utilities import expensive_tests

# Ensure all the plugins are available during testing.
load_plugins()

_FAKE_SECRETS = fake_all_secrets()


def ensure_public_dependencies(dependencies):
    """Some tests are defined with dependencies that require an auth token to download them.
    In this test context, we substitute public files instead."""
    for k, d in dependencies.items():
        if isinstance(d, WebData):
            new_dependency = WebData(source_url=demo_prompt_set_url(d.source_url), headers=None)
            dependencies[k] = new_dependency
    return dependencies


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
    test = TESTS.make_instance(test_name, secrets=_FAKE_SECRETS)

    # TODO remove when localized files are handled better
    # workaround
    if isinstance(test, BaseSafeTestVersion1) and test.locale != EN_US:
        return

    if isinstance(test, PromptResponseTest):
        test_data_path = os.path.join(shared_run_dir, test.__class__.__name__)
        dependencies = ensure_public_dependencies(test.get_dependencies())
        dependency_helper = FromSourceDependencyHelper(
            test_data_path,
            dependencies,
            required_versions={},
        )

        test_items = test.make_test_items(dependency_helper)
        assert len(test_items) > 0


@pytest.mark.parametrize("sut_name", [key for key, _ in SUTS.items()])
def test_all_suts_construct_and_record_init(sut_name):
    sut = SUTS.make_instance(sut_name, secrets=_FAKE_SECRETS)
    assert hasattr(sut, "initialization_record"), "SUT is probably missing @modelgauge_sut() decorator."
    assert isinstance(sut.initialization_record, InitializationRecord)


SUTS_THAT_WE_DONT_CARE_ABOUT_FAILING = {
    "StripedHyena-Nous-7B",
    "olmo-7b-0724-instruct-hf",
    "olmo-2-1124-7b-instruct-hf",
    # mistral-nemo-instruct-2407-hf removed from test because of a bug in HF's date parsing code
    # https://github.com/huggingface/huggingface_hub/issues/2671
    # Remove mistral-nemo-instruct-2407-hf from this set once it's fixed.
    "mistral-nemo-instruct-2407-hf",
    # old, unused
    "qwen2-5-7b-instruct-hf",
    # out of nvidia credits, and maybe they don't want to test these anymore
    "nvidia-nemotron-mini-4b-instruct",
    "nvidia-nemotron-4-340b-instruct",
    "nvidia-llama-3.1-nemotron-70b-instruct",
    "nvidia-mistral-nemo-minitron-8b-8k-instruct",
}


# This test can take a while, and we don't want a test run to fail
# just because an external service is being slow. So we set a somewhat
# high timeout value that gives the test a chance to complete most of the time,
# but still fails if the external service really is flaky or slow, so we can
# get a sense of a real user's experience.
@expensive_tests
@pytest.mark.timeout(HUGGING_FACE_TIMEOUT + 45)  # Hugging Face spinup, plus some time for the test itself
@pytest.mark.parametrize("sut_name", set(SUTS.keys()) - SUTS_THAT_WE_DONT_CARE_ABOUT_FAILING)
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
    assert response.text.strip() != ""


@expensive_tests
@pytest.mark.timeout(HUGGING_FACE_TIMEOUT + 45)  # Hugging Face spinup, plus some time for the test itself
@pytest.mark.parametrize("sut_name", set(SUTS.keys()) - SUTS_THAT_WE_DONT_CARE_ABOUT_FAILING)
def test_can_cache_all_sut_responses(sut_name, tmpdir):
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
    try:
        native_response = sut.evaluate(native_request)
    except Exception as e:
        pytest.skip("SUT failed to evaluate request.")

    with SqlDictCache(tmpdir, "sut_name") as cache:
        assert cache._can_encode(native_request)
        assert cache._can_encode(native_response)
        cache.update_cache(native_request, native_response)
        assert cache.get_cached_response(native_request) == native_response
