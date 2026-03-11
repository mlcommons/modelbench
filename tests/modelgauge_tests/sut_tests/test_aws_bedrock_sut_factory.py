import pytest
from unittest.mock import patch

from modelgauge.dynamic_sut_factory import ModelNotSupportedError
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.aws_bedrock_client import AmazonBedrockSut
from modelgauge.suts.aws_bedrock_sut_factory import AWSBedrockSUTFactory


@pytest.fixture
def factory():
    return AWSBedrockSUTFactory({"aws": {"access_key_id": "value", "secret_access_key": "value"}})


@pytest.fixture
def mock_list_foundation_models():
    models = {
        "modelSummaries": [
            {"modelId": "amazon.nova-1.0-micro-v1:0", "modelLifecycle": "ACTIVE"},
            {"modelId": "old_model", "modelLifecycle": "LEGACY"},
        ]
    }

    with patch("boto3.client") as mock_client:

        mock_client.return_value.list_foundation_models.return_value = models

        yield mock_client


def test_convert_model_id(factory):
    definition = factory._convert_model_id("amazon.nova-v1")
    assert definition.get("maker") == "amazon"
    assert definition.get("model") == "nova-v1"
    assert definition.get("driver") == "aws"

    # Sometimes they have colons
    definition = factory._convert_model_id("amazon.nova-v1:0")
    assert definition.get("maker") == "amazon"
    assert definition.get("model") == "nova-v1.0"
    assert definition.get("driver") == "aws"

    # "." in the model name
    definition = factory._convert_model_id("moonshotai.kimi-k2.5")
    assert definition.get("maker") == "moonshotai"
    assert definition.get("model") == "kimi-k2.5"
    assert definition.get("driver") == "aws"


def test_make_sut(factory, mock_list_foundation_models):
    sut_definition = SUTDefinition(model="nova-1.0-micro-v1.0", maker="amazon", driver="aws")
    sut = factory.make_sut(sut_definition)

    assert isinstance(sut, AmazonBedrockSut)
    assert sut.uid == "amazon/nova-1.0-micro-v1.0:aws"
    assert sut.model_id == "amazon.nova-1.0-micro-v1:0"


def test_make_sut_no_model(factory, mock_list_foundation_models):
    sut_definition = SUTDefinition(model="unknown", maker="amazon", driver="aws")
    with pytest.raises(ModelNotSupportedError):
        factory.make_sut(sut_definition)


def test_make_sut_legacy_model(factory, mock_list_foundation_models):
    sut_definition = SUTDefinition(model="old_model", maker="amazon", driver="aws")
    with pytest.raises(ModelNotSupportedError):
        factory.make_sut(sut_definition)
