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
    return OpenAISUTFactory(raw_secrets={"openai": {"api_key": "some_key"}})


@pytest.fixture
def openai_generic_factory():
    return OpenAIGenericSUTFactory(raw_secrets={"demo": {"api_key": "some_key"}}, base_url="some_url")


@pytest.fixture
def factory():
    return OpenAICompatibleSUTFactory(
        raw_secrets={
            "openai": {"api_key": "some_key", "organization": "some_org"},
            "demo": {"api_key": "some_key"},
        }
    )


@pytest.fixture
def sut_metadata():
    sut_metadata = DynamicSUTMetadata(model="some_model", maker="some_maker", driver="openai", provider="demo")
    return sut_metadata


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


def test_make_sut_with_no_maker(openai_factory):
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
        sut_metadata = DynamicSUTMetadata(model="bogus", maker="openai", driver="openai")
        with pytest.raises(ModelNotSupportedError):
            openai_factory.make_sut(sut_metadata)


### SUTs using the OpenAI client running anywhere
def test_make_generic_sut(openai_generic_factory, sut_metadata):
    openai_generic_factory.base_url = "https://example.com"
    sut = openai_generic_factory.make_sut(sut_metadata)
    assert isinstance(sut, OpenAIChat)
    assert sut.uid == "some_maker/some_model:demo:openai"
    assert sut.model == "some_model"
    assert isinstance(sut.client, OpenAI)


### SUTs using the OpenAI client running anywhere
def test_make_generic_sut_with_late_base_url(openai_generic_factory, sut_metadata):
    sut = openai_generic_factory.make_sut(sut_metadata, base_url="https://example.com")
    assert isinstance(sut, OpenAIChat)
    assert sut.uid == "some_maker/some_model:demo:openai"
    assert sut.model == "some_model"
    assert isinstance(sut.client, OpenAI)


### Factory that decides which kind of OpenAI-compatible SUT you want
def test_factory_makes_the_right_generic_sut(factory, sut_metadata):
    sut = factory.make_sut(sut_metadata, base_url="https://example.org")
    assert isinstance(sut, OpenAIChat)
    assert sut.uid == "some_maker/some_model:demo:openai"
    assert sut.model == "some_model"
    assert isinstance(sut.client, OpenAI)


def test_factory_makes_the_right_openai_sut(factory):
    with patch("modelgauge.suts.openai_sut_factory.OpenAICompatibleSUTFactory._make_client"):
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
