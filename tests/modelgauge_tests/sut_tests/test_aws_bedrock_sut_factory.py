import pytest
from unittest.mock import patch

from modelgauge.dynamic_sut_factory import ModelNotSupportedError
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.aws_bedrock_client import AmazonNovaSut
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
