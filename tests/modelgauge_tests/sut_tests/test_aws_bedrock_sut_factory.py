import pytest
from unittest.mock import patch

from modelgauge.dynamic_sut_factory import ModelNotSupportedError
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.aws_bedrock_client import AmazonBedrockSut
from modelgauge.suts.aws_bedrock_sut_factory import AWSBedrockSUTFactory


@pytest.fixture
def factory():
    return AWSBedrockSUTFactory({"aws": {"access_key_id": "value", "secret_access_key": "value"}})


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

