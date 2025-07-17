from unittest.mock import patch

import pytest

from modelgauge.config import load_secrets_from_config
from modelgauge.dynamic_sut_factory import ModelNotSupportedError
from modelgauge.dynamic_sut_metadata import DynamicSUTMetadata
from modelgauge.suts.together_client import TogetherChatSUT
from modelgauge.suts.together_sut_factory import TogetherSUTFactory
from modelgauge_tests.utilities import expensive_tests


@pytest.fixture
def factory():
    return TogetherSUTFactory({"together": {"api_key": "value"}})


def test_make_sut(factory):
    with patch("modelgauge.suts.together_sut_factory.TogetherSUTFactory._find", return_value="google/gemma:together"):
        sut_metadata = DynamicSUTMetadata(model="gemma", maker="google", driver="together")
        sut = factory.make_sut(sut_metadata)
        assert isinstance(sut, TogetherChatSUT)
        assert sut.uid == "google/gemma:together"
        assert sut.model == "google/gemma"
        assert sut.api_key == "value"


def test_make_sut_bad_model(factory):
    sut_metadata = DynamicSUTMetadata(model="bogus", maker="fake", driver="together")
    with patch("modelgauge.suts.together_sut_factory.TogetherSUTFactory._find", side_effect=ModelNotSupportedError()):
        with pytest.raises(ModelNotSupportedError):
            _ = factory.make_sut(sut_metadata)


def test_find(factory):
    with patch(
        "modelgauge.suts.together_sut_factory.together.Models.list",
        return_value=[{"id": "google/gemma"}],
    ):
        sut_metadata = DynamicSUTMetadata(model="gemma", maker="google", driver="together")
        assert factory._find(sut_metadata) == sut_metadata.external_model_name()


def test_find_bad_model(factory):
    sut_metadata = DynamicSUTMetadata(model="any", maker="any", driver="together")
    with patch(
        "modelgauge.suts.together_sut_factory.together.Models.list",
        return_value=None,
    ):
        with pytest.raises(ModelNotSupportedError):
            _ = factory._find(sut_metadata)


@expensive_tests
def test_connection():
    factory = TogetherSUTFactory(load_secrets_from_config())
    sut_metadata = DynamicSUTMetadata(model="meta-llama", maker="Llama-3-70b-chat-hf", driver="together")
    sut = factory.make_sut(sut_metadata)
    assert sut.uid == "meta-llama/Llama-3-70b-chat-hf:together"
    assert sut.model == "meta-llama/Llama-3-70b-chat-hf"
