from collections import namedtuple
from unittest.mock import patch, MagicMock

import pytest

from modelgauge.dynamic_sut_factory import ModelNotSupportedError
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.mistral_sut import MistralAISut
from modelgauge.suts.mistral_sut_factory import MistralSUTFactory


@pytest.fixture
def factory():
    return MistralSUTFactory({"mistralai": {"api_key": "value"}})


def test_make_sut(factory):
    with patch("modelgauge.suts.mistral_client.MistralAIClient.model_info", return_value="model exists"):
        sut_definition = SUTDefinition(model="bar", maker="foo", driver="mistral")
        sut = factory.make_sut(sut_definition)

        assert isinstance(sut, MistralAISut)
        assert sut.uid == "foo/bar:mistral"
        assert sut.model_name == "foo/bar"
        assert sut._api_key.value == "value"


def test_make_sut_bad_model(factory):
    sut_definition = SUTDefinition(model="bogus", maker="fake", driver="mistral")
    with patch("modelgauge.suts.mistral_client.MistralAIClient.model_info", side_effect=Exception()):
        with pytest.raises(ModelNotSupportedError):
            factory.make_sut(sut_definition)


def test_list_suts(factory):
    model_list = MagicMock()
    model_list.data = [(namedtuple("FakeModel", ["id"])("thingy-1.0"))]
    factory._client = MagicMock()
    factory._client.client.models.list.return_value = model_list
    assert "mistral/thingy-1.0" in factory.list_suts()
