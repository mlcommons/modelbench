from unittest.mock import patch

import pytest
from openai import OpenAI

from modelgauge.config import load_secrets_from_config
from modelgauge.dynamic_sut_factory import ModelNotSupportedError, ProviderNotFoundError
from modelgauge.sut_definition import SUTDefinition
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
def sut_definition():
    return SUTDefinition(model="some_model", maker="some_maker", driver="openai", provider="demo")


### OpenAI SUTs running on OpenAI
def test_make_sut(openai_factory):
    with patch(
        "modelgauge.suts.openai_sut_factory.OpenAISUTFactory._model_exists",
        return_value=True,
    ):
        sut_definition = SUTDefinition(model="gpt-4o", maker="openai", driver="openai")
        sut = openai_factory.make_sut(sut_definition)
    assert isinstance(sut, OpenAIChat)
    assert sut.uid == "openai/gpt-4o:openai"
    assert sut.model == "gpt-4o"
    assert isinstance(sut.client, OpenAI)


def test_make_sut_with_no_maker(openai_factory):
    with patch(
        "modelgauge.suts.openai_sut_factory.OpenAISUTFactory._model_exists",
        return_value=True,
    ):
        sut_definition = SUTDefinition(model="gpt-4o", driver="openai")
        sut = openai_factory.make_sut(sut_definition)
    assert isinstance(sut, OpenAIChat)
    assert sut.uid == "gpt-4o:openai"
    assert sut.model == "gpt-4o"


def test_make_unknown_sut_raises_error(openai_factory):
    with patch(
        "modelgauge.suts.openai_sut_factory.OpenAISUTFactory._model_exists",
        return_value=False,
    ):
        sut_definition = SUTDefinition(model="bogus", maker="openai", driver="openai")
        with pytest.raises(ModelNotSupportedError):
            openai_factory.make_sut(sut_definition)


### SUTs using the OpenAI client running anywhere
def test_make_generic_sut(openai_generic_factory, sut_definition):
    openai_generic_factory.base_url = "https://example.com"
    sut = openai_generic_factory.make_sut(sut_definition)
    assert isinstance(sut, OpenAIChat)
    assert sut.uid == "some_maker/some_model:demo:openai"
    assert sut.model == "some_model"
    assert isinstance(sut.client, OpenAI)


### SUTs using the OpenAI client running anywhere
def test_make_generic_sut_with_late_base_url(openai_generic_factory, sut_definition):
    sut = openai_generic_factory.make_sut(sut_definition, base_url="https://example.com")
    assert isinstance(sut, OpenAIChat)
    assert sut.uid == "some_maker/some_model:demo:openai"
    assert sut.model == "some_model"
    assert isinstance(sut.client, OpenAI)


### Factory that decides which kind of OpenAI-compatible SUT you want
def test_factory_makes_the_right_generic_sut(factory):
    sut_definition = SUTDefinition(
        model="some_model", maker="some_maker", driver="openai", provider="demo", base_url="https://example.org"
    )
    sut = factory.make_sut(sut_definition)
    assert isinstance(sut, OpenAIChat)
    assert sut.uid == "some_maker/some_model:demo:openai;url=https://example.org"
    assert sut.model == "some_model"
    assert isinstance(sut.client, OpenAI)


### Factory that tries to create a generic OpenAI-compatible SUT if you pass in the URL and secrets exist
def test_factory_tries_to_make_a_generic_sut(factory, sut_definition):
    # there's a base URL but no secret for that provider, so we give up
    sut_definition = SUTDefinition(
        model="some_model", maker="some_maker", driver="openai", provider="unknown", base_url="https://www.example.net"
    )
    with pytest.raises(ProviderNotFoundError):
        _ = factory.make_sut(sut_definition)

    # there is a secret for the "unknown" provider, so we try to make a SUT
    new_factory = OpenAICompatibleSUTFactory(raw_secrets={"unknown": {"api_key": "some_key"}})
    sut = new_factory.make_sut(sut_definition)
    assert isinstance(sut, OpenAIChat)
    assert isinstance(sut.client, OpenAI)


def test_factory_makes_the_right_openai_sut(factory):
    with patch("modelgauge.suts.openai_sut_factory.OpenAICompatibleSUTFactory._make_client"):
        sut_definition = SUTDefinition(model="gpt-5", maker="openai", driver="openai")
        sut = factory.make_sut(sut_definition)
        assert sut.uid == "openai/gpt-5:openai"
        assert sut.model == "gpt-5"


@expensive_tests
def test_connection():
    factory = OpenAISUTFactory(load_secrets_from_config(path="."))
    sut_definition = SUTDefinition(model="gpt-4o", maker="openai", driver="openai")
    sut = factory.make_sut(sut_definition)
    assert sut.uid == "openai/gpt-4o:openai"
    assert sut.model == "gpt-4o"
