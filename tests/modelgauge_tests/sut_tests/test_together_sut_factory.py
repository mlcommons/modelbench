from unittest.mock import MagicMock

import pytest

from modelgauge.config import load_secrets_from_config
from modelgauge.dynamic_sut_factory import ModelNotSupportedError
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.together_client import TogetherChatSUT, TogetherDedicatedChatSUT
from modelgauge.suts.together_sut_factory import TogetherSUTFactory
from modelgauge_tests.utilities import expensive_tests


@pytest.fixture
def together_factory():
    factory = TogetherSUTFactory({"together": {"api_key": "value", "project_id": "value"}})
    factory.client = MagicMock()
    return factory


def test_find_serverless(together_factory):
    together_factory.client.chat.completions.create.return_value = {}
    result = together_factory._find_serverless("google/gemma")
    assert result == "google/gemma"
    together_factory.client.chat.completions.create.assert_called_once_with(
        model="google/gemma",
        messages=[{"role": "user", "content": "Anybody home?"}],
    )


def test_find_serverless_bad_model(together_factory):
    together_factory.client.chat.completions.create.side_effect = Exception("Model not available")
    result = together_factory._find_serverless("google/gemma")
    assert result is None


def test_find_dedicated(together_factory):
    mock_endpoint = MagicMock()
    mock_endpoint.model = "google/gemma"
    mock_endpoint.name = "my-dedicated-endpoint"
    together_factory.client.endpoints.list.return_value.data = [mock_endpoint]

    result = together_factory._find_dedicated("google/gemma")
    assert result == "my-dedicated-endpoint"
    together_factory.client.endpoints.list.assert_called_once_with(type="dedicated", mine=True)


def test_find_dedicated_not_found(together_factory):
    together_factory.client.endpoints.list.return_value.data = []
    result = together_factory._find_dedicated("google/gemma")
    assert result is None


def test_find_dedicated_error(together_factory):
    together_factory.client.endpoints.list.side_effect = Exception("API error")
    result = together_factory._find_dedicated("google/gemma")
    assert result is None


def test_make_sut_serverless(together_factory):
    together_factory.client.chat.completions.create.return_value = {}
    sut_definition = SUTDefinition(model="gemma", maker="google", driver="together")
    sut = together_factory.make_sut(sut_definition)
    assert isinstance(sut, TogetherChatSUT)
    assert sut.uid == "google/gemma:together"
    assert sut.model == "google/gemma"
    assert sut.api_key == "value"


def test_make_sut_falls_back_to_dedicated(together_factory):
    together_factory.client.chat.completions.create.side_effect = Exception("not serverless")
    mock_endpoint = MagicMock()
    mock_endpoint.model = "google/gemma"
    mock_endpoint.name = "my-dedicated-endpoint"
    together_factory.client.endpoints.list.return_value.data = [mock_endpoint]

    sut_definition = SUTDefinition(model="gemma", maker="google", driver="together")
    sut = together_factory.make_sut(sut_definition)
    assert isinstance(sut, TogetherDedicatedChatSUT)
    assert sut.uid == "google/gemma:together"
    assert sut.model == "my-dedicated-endpoint"
    assert sut.api_key == "value"


def test_make_sut_not_found(together_factory):
    together_factory.client.chat.completions.create.side_effect = Exception("not serverless")
    together_factory.client.endpoints.list.return_value.data = []

    sut_definition = SUTDefinition(model="bogus", maker="fake", driver="together")
    with pytest.raises(ModelNotSupportedError):
        together_factory.make_sut(sut_definition)


@expensive_tests
def test_serverless_connection():
    factory = TogetherSUTFactory(load_secrets_from_config(path="."))
    sut_definition = SUTDefinition(maker="meta-llama", model="Llama-3.3-70B-Instruct-Turbo", driver="together")
    sut = factory.make_sut(sut_definition)
    assert sut.uid == "meta-llama/llama-3.3-70b-instruct-turbo:together"
    assert sut.model == "meta-llama/Llama-3.3-70B-Instruct-Turbo"
