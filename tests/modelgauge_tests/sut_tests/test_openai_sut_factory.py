import pytest
from unittest.mock import patch

from openai import OpenAI

from modelgauge.config import load_secrets_from_config
from modelgauge.dynamic_sut_factory import ModelNotSupportedError
from modelgauge.dynamic_sut_metadata import DynamicSUTMetadata
from modelgauge.suts.openai_client import OpenAIChat
from modelgauge.suts.openai_sut_factory import OpenAICompatibleSUTFactory, OpenAIGenericSUTFactory, OpenAISUTFactory
from modelgauge_tests.utilities import expensive_tests


@pytest.fixture
def openai_factory():
    return OpenAISUTFactory({"openai": {"api_key": "value"}})


@pytest.fixture
def openai_generic_factory():
    return OpenAIGenericSUTFactory({"some_vendor": {"api_key": "value", "base_url": "some_url"}})


@pytest.fixture
def factory():
    return OpenAICompatibleSUTFactory(
        {
            "openai": {"api_key": "value", "organization": "some_org"},
            "some_vendor": {"api_key": "some_key", "base_url": "some_url"},
        }
    )


### OpenAI SUTs running on OpenAI
def test_make_sut(openai_factory):
    with patch(
        "modelgauge.suts.openai_sut_factory.OpenAISUTFactory._model_exists",
        return_value=True,
    ):
        sut_metadata = DynamicSUTMetadata(model="gpt-4o", maker="openai", driver="openai")
        sut = openai_factory.make_sut(sut_metadata)
    assert isinstance(sut, OpenAIChat)
    assert sut.uid == "openai/gpt-4o:openai"
    assert sut.model == "gpt-4o"
    assert isinstance(sut.client, OpenAI)


def test_make_sut_no_maker(openai_factory):
    with patch(
        "modelgauge.suts.openai_sut_factory.OpenAISUTFactory._model_exists",
        return_value=True,
    ):
        sut_metadata = DynamicSUTMetadata(model="gpt-4o", driver="openai")
        sut = openai_factory.make_sut(sut_metadata)
    assert isinstance(sut, OpenAIChat)
    assert sut.uid == "gpt-4o:openai"
    assert sut.model == "gpt-4o"


def test_make_unknown_sut_raises_error(openai_factory):
    with patch(
        "modelgauge.suts.openai_sut_factory.OpenAISUTFactory._model_exists",
        return_value=False,
    ):
        sut_metadata = DynamicSUTMetadata(model="gpt-4o", maker="openai", driver="openai")
        with pytest.raises(ModelNotSupportedError):
            openai_factory.make_sut(sut_metadata)


### SUTs using the OpenAI client running anywhere
def test_make_generic_sut(openai_generic_factory):
    sut_metadata = DynamicSUTMetadata(model="gemini-something", maker="google", driver="openai", provider="some_vendor")
    sut = openai_generic_factory.make_sut(sut_metadata)
    assert isinstance(sut, OpenAIChat)
    assert sut.uid == "google/gemini-something:some_vendor:openai"
    assert sut.model == "gemini-something"
    assert isinstance(sut.client, OpenAI)


### Factory that decides which kind of OpenAI-compatible SUT you want
def test_factory_makes_the_right_generic_sut(factory):
    sut_metadata = DynamicSUTMetadata(model="gemini-something", maker="google", driver="openai", provider="some_vendor")
    sut = factory.make_sut(sut_metadata)
    assert isinstance(sut, OpenAIChat)
    assert sut.uid == "google/gemini-something:some_vendor:openai"
    assert sut.model == "gemini-something"
    assert isinstance(sut.client, OpenAI)


def test_factory_makes_the_right_openai_sut(factory):
    with patch("modelgauge.suts.openai_sut_factory.OpenAISUTFactory.client"):
        sut_metadata = DynamicSUTMetadata(model="gpt-5", maker="openai", driver="openai")
        sut = factory.make_sut(sut_metadata)
        assert sut.uid == "openai/gpt-5:openai"
        assert sut.model == "gpt-5"


@expensive_tests
def test_connection():
    factory = OpenAISUTFactory(load_secrets_from_config(path="."))
    sut_metadata = DynamicSUTMetadata(model="gpt-4o", maker="openai", driver="openai")
    sut = factory.make_sut(sut_metadata)
    assert sut.uid == "openai/gpt-4o:openai"
    assert sut.model == "gpt-4o"
