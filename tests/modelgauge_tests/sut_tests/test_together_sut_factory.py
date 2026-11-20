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


def test_serverless_list_suts(serverless_factory):
    # Too many to list
    assert serverless_factory.list_suts() is None


def test_dedicated_make_sut(dedicated_factory, mocker):
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
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": []}
    mocker.patch("modelgauge.suts.together_client._retrying_request", return_value=mock_response)

    sut_definition = SUTDefinition(model="bogus", maker="fake", driver="together-dedicated")
    with pytest.raises(ModelNotSupportedError):
        dedicated_factory.make_sut(sut_definition)


def test_dedicated_list_suts(dedicated_factory, mocker):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {
                "id": "ep-1",
                "deployments": [
                    {"id": "dep-1", "name": "google/gemma"},
                    {"id": "dep-2", "name": "gpt-oss-20b"},
                ],
            },
            {
                "id": "ep-2",
                "deployments": [
                    {"id": "dep-3", "name": "google/gemma"},
                    {"id": "dep-4", "name": ""},
                    {"id": "dep-5"},
                ],
            },
        ]
    }
    request = mocker.patch(
        "modelgauge.suts.together_sut_factory._retrying_request",
        return_value=mock_response,
    )

    suts = dedicated_factory.list_suts()

    assert [sut.uid for sut in suts] == [
        "google/gemma:together-dedicated",
        "gpt-oss-20b:together-dedicated",
    ]
    request.assert_called_once_with(
        "https://api.together.ai/v2/projects/value/endpoints",
        {
            "accept": "application/json",
            "authorization": "Bearer value",
        },
        None,
        "GET",
    )


@expensive_tests
def test_serverless_connection():
    factory = TogetherServerlessSUTFactory(load_secrets_from_config(path="."))
    sut_definition = SUTDefinition(
        maker="meta-llama", model="Llama-3.3-70B-Instruct-Turbo", driver="together-serverless"
    )
    sut = factory.make_sut(sut_definition)
    assert sut.uid == "meta-llama/llama-3.3-70b-instruct-turbo:together-serverless"
    assert sut.model == "meta-llama/Llama-3.3-70B-Instruct-Turbo"
