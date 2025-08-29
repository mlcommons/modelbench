from unittest.mock import patch, MagicMock

import pytest

from modelgauge.config import load_secrets_from_config
from modelgauge.dynamic_sut_factory import ModelNotSupportedError
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.together_client import TogetherChatSUT
from modelgauge.suts.together_sut_factory import TogetherSUTFactory
from modelgauge_tests.utilities import expensive_tests


@pytest.fixture
def factory():
    return TogetherSUTFactory({"together": {"api_key": "value"}})


def test_make_sut(factory):
    with patch("modelgauge.suts.together_sut_factory.TogetherSUTFactory._find", return_value="google/gemma:together"):
        sut_definition = SUTDefinition(model="gemma", maker="google", driver="together")
        sut = factory.make_sut(sut_definition)
        assert isinstance(sut, TogetherChatSUT)
        assert sut.uid == "google/gemma:together"
        assert sut.model == "google/gemma"
        assert sut.api_key == "value"


def test_make_sut_bad_model(factory):
    sut_definition = SUTDefinition(model="bogus", maker="fake", driver="together")
    with patch("modelgauge.suts.together_sut_factory.TogetherSUTFactory._find", side_effect=ModelNotSupportedError()):
        with pytest.raises(ModelNotSupportedError):
            _ = factory.make_sut(sut_definition)


def test_find(factory):
    mock_together = MagicMock()
    mock_together.return_value.chat.completions.create.return_value = {}  # The method doesn't use the return value.
    with patch("modelgauge.suts.together_sut_factory.Together", mock_together):
        sut_definition = SUTDefinition(model="gemma", maker="google", driver="together")
        assert factory._find(sut_definition) == sut_definition.external_model_name()


def test_find_bad_model(factory):
    sut_definition = SUTDefinition(model="any", maker="any", driver="together")
    mock_together = MagicMock()
    mock_together.return_value.chat.completions.create.side_effect = Exception("Model not available")
    with patch("modelgauge.suts.together_sut_factory.Together", mock_together):
        with pytest.raises(ModelNotSupportedError):
            _ = factory._find(sut_definition)


@expensive_tests
def test_connection():
    factory = TogetherSUTFactory(load_secrets_from_config(path="."))
    sut_definition = SUTDefinition(maker="meta-llama", model="Llama-3-70b-chat-hf", driver="together")
    sut = factory.make_sut(sut_definition)
    assert sut.uid == "meta-llama/llama-3-70b-chat-hf:together"
    assert sut.model == "meta-llama/Llama-3-70b-chat-hf"
