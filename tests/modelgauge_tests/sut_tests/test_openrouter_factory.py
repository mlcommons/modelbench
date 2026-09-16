from unittest.mock import MagicMock

import pytest
from openai import OpenAI

from modelgauge.dynamic_sut_factory import ModelNotSupportedError
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.openrouter_sut_factory import OPENROUTER_BASE_URL, OpenRouterSUT, OpenRouterSUTFactory
from modelgauge_tests.utilities import FakeObject


@pytest.fixture
def factory():
    factory = OpenRouterSUTFactory(
        raw_secrets={
            "openrouter": {"api_key": "some_key"},
        }
    )
    real_client = OpenAI(api_key="some_key", base_url=OPENROUTER_BASE_URL)
    list_response = MagicMock()
    list_response.data = [
        FakeObject(id="qwen/qwen3-max-0902", provider="alibaba"),
        FakeObject(id="meta-llama/llama-3.1-70b", provider="meta"),
    ]
    real_client.models.list = MagicMock(return_value=list_response)
    factory._client = real_client
    return factory


def test_factory_makes_correct_openrouter_sut(factory):
    sut_definition = SUTDefinition(
        maker="qwen",
        model="qwen3-max-0902",
        provider="alibaba",
        driver="openrouter",
    )
    sut = factory.make_sut(sut_definition)

    assert isinstance(sut, OpenRouterSUT)
    assert sut.uid == "qwen/qwen3-max-0902:alibaba:openrouter"
    assert sut.model == "qwen/qwen3-max-0902"
    assert sut.client is factory.client
    assert str(factory.client.base_url).startswith(OPENROUTER_BASE_URL)
    factory.client.models.list.assert_called()


def test_make_sut_bad_model(factory):
    sut_definition = SUTDefinition(
        maker="qwen",
        model="bogus",
        provider="alibaba",
        driver="openrouter",
    )
    with pytest.raises(ModelNotSupportedError):
        factory.make_sut(sut_definition)


def test_list_suts(factory):
    suts = factory.list_suts()
    assert suts is not None
    uids = [s.uid for s in suts]
    assert "qwen/qwen3-max-0902:openrouter" in uids
    assert "meta-llama/llama-3.1-70b:openrouter" in uids
