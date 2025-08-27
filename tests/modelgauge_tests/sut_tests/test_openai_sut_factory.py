import pytest
from unittest.mock import patch

from modelgauge.config import load_secrets_from_config
from modelgauge.dynamic_sut_factory import ModelNotSupportedError
from modelgauge.dynamic_sut_metadata import DynamicSUTMetadata
from modelgauge.suts.openai_client import OpenAIChat
from modelgauge.suts.openai_sut_factory import OpenAISUTFactory
from modelgauge_tests.utilities import expensive_tests


@pytest.fixture
def factory():
    return OpenAISUTFactory({"openai": {"api_key": "value"}})


def test_make_sut(factory):
    with patch(
        "modelgauge.suts.openai_sut_factory.OpenAISUTFactory._model_exists",
        return_value=True,
    ):
        sut_metadata = DynamicSUTMetadata(model="gpt-4o", maker="openai", driver="openai")
        sut = factory.make_sut(sut_metadata)
    assert isinstance(sut, OpenAIChat)
    assert sut.uid == "openai/gpt-4o:openai"
    assert sut.model == "gpt-4o"
    assert sut.api_key == "value"


def test_make_sut_no_maker(factory):
    with patch(
        "modelgauge.suts.openai_sut_factory.OpenAISUTFactory._model_exists",
        return_value=True,
    ):
        sut_metadata = DynamicSUTMetadata(model="gpt-4o", driver="openai")
        sut = factory.make_sut(sut_metadata)
    assert isinstance(sut, OpenAIChat)
    assert sut.uid == "gpt-4o:openai"
    assert sut.model == "gpt-4o"


def test_make_unknown_sut_raises_error(factory):
    with patch(
        "modelgauge.suts.openai_sut_factory.OpenAISUTFactory._model_exists",
        return_value=False,
    ):
        sut_metadata = DynamicSUTMetadata(model="gpt-4o", maker="openai", driver="openai")
        with pytest.raises(ModelNotSupportedError):
            factory.make_sut(sut_metadata)


@expensive_tests
def test_connection():
    factory = OpenAISUTFactory(load_secrets_from_config(path="."))
    sut_metadata = DynamicSUTMetadata(model="gpt-4o", maker="openai", driver="openai")
    sut = factory.make_sut(sut_metadata)
    assert sut.uid == "openai/gpt-4o:openai"
    assert sut.model == "gpt-4o"
