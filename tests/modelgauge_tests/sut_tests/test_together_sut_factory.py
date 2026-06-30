from unittest.mock import patch, MagicMock

import pytest

from modelgauge.config import load_secrets_from_config
from modelgauge.dynamic_sut_factory import ModelNotSupportedError
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.together_client import TogetherChatSUT, TogetherDedicatedChatSUT
from modelgauge.suts.together_sut_factory import TogetherDedicatedSUTFactory, TogetherServerlessSUTFactory, TogetherSUTFactory
from modelgauge_tests.utilities import expensive_tests


@pytest.fixture
def serverless_factory():
    return TogetherServerlessSUTFactory({"together": {"api_key": "value"}})


@pytest.fixture
def dedicated_factory():
    return TogetherDedicatedSUTFactory({"together": {"api_key": "value"}})


def test_make_sut(serverless_factory):
    with patch("modelgauge.suts.together_sut_factory.TogetherServerlessSUTFactory._find", return_value="google/gemma:together"):
        sut_definition = SUTDefinition(model="gemma", maker="google", driver="together")
        sut = serverless_factory.make_sut(sut_definition)
        assert isinstance(sut, TogetherChatSUT)
        assert sut.uid == "google/gemma:together"
        assert sut.model == "google/gemma"
        assert sut.api_key == "value"


def test_make_sut_bad_model(serverless_factory):
    sut_definition = SUTDefinition(model="bogus", maker="fake", driver="together")
    with patch("modelgauge.suts.together_sut_factory.TogetherServerlessSUTFactory._find", side_effect=ModelNotSupportedError()):
        with pytest.raises(ModelNotSupportedError):
            _ = serverless_factory.make_sut(sut_definition)


def test_find(serverless_factory):
    mock_together = MagicMock()
    mock_together.return_value.chat.completions.create.return_value = {}  # The method doesn't use the return value.
    with patch("modelgauge.suts.together_sut_factory.Together", mock_together):
        sut_definition = SUTDefinition(model="gemma", maker="google", driver="together")
        assert serverless_factory._find(sut_definition) == sut_definition.external_model_name()


def test_find_bad_model(serverless_factory):
    sut_definition = SUTDefinition(model="any", maker="any", driver="together")
    mock_together = MagicMock()
    mock_together.return_value.chat.completions.create.side_effect = Exception("Model not available")
    with patch("modelgauge.suts.together_sut_factory.Together", mock_together):
        with pytest.raises(ModelNotSupportedError):
            _ = serverless_factory._find(sut_definition)


@expensive_tests
def test_serverless_connection():
    factory = TogetherServerlessSUTFactory(load_secrets_from_config(path="."))
    sut_definition = SUTDefinition(maker="meta-llama", model="Llama-3.3-70B-Instruct-Turbo", driver="together")
    sut = factory.make_sut(sut_definition)
    assert sut.uid == "meta-llama/llama-3.3-70b-instruct-turbo:together"
    assert sut.model == "meta-llama/Llama-3.3-70B-Instruct-Turbo"


def test_make_sut_dedicated_not_found(dedicated_factory):
    mock_together = MagicMock()
    mock_together.return_value.endpoints.list.return_value.data = []

    with patch("modelgauge.suts.together_sut_factory.Together", mock_together):
        sut_definition = SUTDefinition(model="gemma", maker="google", driver="together")
        with pytest.raises(ModelNotSupportedError):
            dedicated_factory.make_sut(sut_definition)


def test_make_sut_dedicated(dedicated_factory):
    mock_endpoint = MagicMock()
    mock_endpoint.model = "google/gemma"
    mock_endpoint.name = "my-dedicated-endpoint"

    mock_together = MagicMock()
    mock_together.return_value.endpoints.list.return_value.data = [mock_endpoint]

    with patch("modelgauge.suts.together_sut_factory.Together", mock_together):
        sut_definition = SUTDefinition(model="gemma", maker="google", driver="together")
        sut = dedicated_factory.make_sut(sut_definition)
        assert isinstance(sut, TogetherDedicatedChatSUT)
        assert sut.uid == "google/gemma:together"
        assert sut.model == "my-dedicated-endpoint"
        assert sut.api_key == "value"
        mock_together.return_value.endpoints.list.assert_called_once_with(type="dedicated", mine=True)


@pytest.fixture
def together_factory():
    return TogetherSUTFactory({"together": {"api_key": "value"}})


def test_together_factory_uses_serverless(together_factory):
    mock_sut = MagicMock(spec=TogetherChatSUT)
    together_factory.serverless_factory.make_sut = MagicMock(return_value=mock_sut)
    together_factory.dedicated_factory.make_sut = MagicMock()

    sut_definition = SUTDefinition(model="gemma", maker="google", driver="together")
    result = together_factory.make_sut(sut_definition)

    assert result is mock_sut
    together_factory.serverless_factory.make_sut.assert_called_once_with(sut_definition)
    together_factory.dedicated_factory.make_sut.assert_not_called()


def test_together_factory_falls_back_to_dedicated(together_factory):
    mock_sut = MagicMock(spec=TogetherDedicatedChatSUT)
    together_factory.serverless_factory.make_sut = MagicMock(side_effect=ModelNotSupportedError())
    together_factory.dedicated_factory.make_sut = MagicMock(return_value=mock_sut)

    sut_definition = SUTDefinition(model="gemma", maker="google", driver="together")
    result = together_factory.make_sut(sut_definition)

    assert result is mock_sut
    together_factory.serverless_factory.make_sut.assert_called_once_with(sut_definition)
    together_factory.dedicated_factory.make_sut.assert_called_once_with(sut_definition)


def test_together_factory_both_fail(together_factory):
    together_factory.serverless_factory.make_sut = MagicMock(side_effect=ModelNotSupportedError())
    together_factory.dedicated_factory.make_sut = MagicMock(side_effect=ModelNotSupportedError())

    sut_definition = SUTDefinition(model="bogus", maker="fake", driver="together")
    with pytest.raises(ModelNotSupportedError):
        together_factory.make_sut(sut_definition)
