from collections import namedtuple
from unittest.mock import MagicMock

import pytest

from modelgauge.config import load_secrets_from_config
from modelgauge.dynamic_sut_factory import ModelNotSupportedError
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.together_client import TogetherChatSUT, TogetherDedicatedChatSUT
from modelgauge.suts.together_sut_factory import (
    TogetherDedicatedSUTFactory,
    TogetherServerlessSUTFactory,
)
from modelgauge_tests.utilities import expensive_tests


@pytest.fixture
def serverless_factory():
    factory = TogetherServerlessSUTFactory({"together": {"api_key": "value", "project_id": "value"}})
    factory.client = MagicMock()
    return factory


@pytest.fixture
def dedicated_factory():
    factory = TogetherDedicatedSUTFactory({"together": {"api_key": "value", "project_id": "value"}})
    factory.client = MagicMock()
    return factory


def test_serverless_find(serverless_factory):
    serverless_factory.client.chat.completions.create.return_value = {}
    result = serverless_factory._find("google/gemma")
    assert result == "google/gemma"
    serverless_factory.client.chat.completions.create.assert_called_once_with(
        model="google/gemma",
        messages=[{"role": "user", "content": "Anybody home?"}],
    )


def test_serverless_find_bad_model(serverless_factory):
    serverless_factory.client.chat.completions.create.side_effect = Exception("Model not available")
    result = serverless_factory._find("google/gemma")
    assert result is None


def test_serverless_make_sut(serverless_factory):
    serverless_factory.client.chat.completions.create.return_value = {}
    sut_definition = SUTDefinition(model="gemma", maker="google", driver="together-serverless")
    sut = serverless_factory.make_sut(sut_definition)
    assert isinstance(sut, TogetherChatSUT)
    assert sut.uid == "google/gemma:together-serverless"
    assert sut.model == "google/gemma"
    assert sut.api_key == "value"


def test_serverless_make_sut_not_found(serverless_factory):
    serverless_factory.client.chat.completions.create.side_effect = Exception("not serverless")
    sut_definition = SUTDefinition(model="bogus", maker="fake", driver="together-serverless")
    with pytest.raises(ModelNotSupportedError):
        serverless_factory.make_sut(sut_definition)


def test_list_suts(serverless_factory):
    m = namedtuple("FakeModel", ["id"])("fnord/thingy-1.0")
    serverless_factory._client = MagicMock()
    serverless_factory._client.models.list.return_value = [m]

    assert serverless_factory.list_suts() == [m.id]

def test_dedicated_make_sut(dedicated_factory, mocker):
    mock_endpoint = MagicMock()
    mock_endpoint.model = "google/gemma"
    mock_endpoint.name = "my-dedicated-endpoint"
    dedicated_factory.client.endpoints.list.return_value.data = [mock_endpoint]

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {
                "id": "ep-id",
                "name": "my-dedicated-endpoint",
                "deployments": [{"id": "dep-id", "name": "google/gemma"}],
            }
        ]
    }
    mocker.patch("modelgauge.suts.together_client._retrying_request", return_value=mock_response)

    sut_definition = SUTDefinition(model="gemma", maker="google", driver="together-dedicated")
    sut = dedicated_factory.make_sut(sut_definition)
    assert isinstance(sut, TogetherDedicatedChatSUT)
    assert sut.uid == "google/gemma:together-dedicated"
    assert sut.model == "my-dedicated-endpoint"
    assert sut.api_key == "value"


def test_dedicated_make_sut_not_found(dedicated_factory, mocker):
    dedicated_factory.client.endpoints.list.return_value.data = []
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": []}
    mocker.patch("modelgauge.suts.together_client._retrying_request", return_value=mock_response)

    sut_definition = SUTDefinition(model="bogus", maker="fake", driver="together-dedicated")
    with pytest.raises(ModelNotSupportedError):
        dedicated_factory.make_sut(sut_definition)


@expensive_tests
def test_serverless_connection():
    factory = TogetherServerlessSUTFactory(load_secrets_from_config(path="."))
    sut_definition = SUTDefinition(
        maker="meta-llama", model="Llama-3.3-70B-Instruct-Turbo", driver="together-serverless"
    )
    sut = factory.make_sut(sut_definition)
    assert sut.uid == "meta-llama/llama-3.3-70b-instruct-turbo:together-serverless"
    assert sut.model == "meta-llama/Llama-3.3-70B-Instruct-Turbo"
