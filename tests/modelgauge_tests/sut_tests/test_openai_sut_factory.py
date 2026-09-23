from unittest.mock import patch

import pytest
from openai import OpenAI

from modelgauge.config import load_secrets_from_config
from modelgauge.dynamic_sut_factory import ModelNotSupportedError
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.openai_client import OpenAIResponsesSUT
from modelgauge.suts.openai_sut_factory import OpenAISUTFactory
from modelgauge_tests.utilities import expensive_tests


@pytest.fixture
def openai_factory():
    return OpenAISUTFactory(raw_secrets={"openai": {"api_key": "some_key"}})


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
    assert isinstance(sut, OpenAIResponsesSUT)
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
    assert isinstance(sut, OpenAIResponsesSUT)
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


@expensive_tests
def test_connection():
    factory = OpenAISUTFactory(load_secrets_from_config(path="."))
    sut_definition = SUTDefinition(model="gpt-4o", maker="openai", driver="openai")
    sut = factory.make_sut(sut_definition)
    assert sut.uid == "openai/gpt-4o:openai"
    assert sut.model == "gpt-4o"
