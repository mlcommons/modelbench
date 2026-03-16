import pytest
from unittest.mock import MagicMock

from modelgauge.dynamic_sut_factory import ModelNotSupportedError
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.nvidia_nim_api_client import NvidiaNIMApiClient
from modelgauge.suts.nvidia_nim_sut_factory import NvidiaNIMSUTFactory


@pytest.fixture
def factory():
    return NvidiaNIMSUTFactory({"nvidia-nim-api": {"api_key": "value"}})


def test_make_sut(factory):
    factory._client = MagicMock()
    factory._client.models.retrieve.return_value = "model exists"

    sut_definition = SUTDefinition(model="bar", maker="foo", driver="nvidia-nim")
    sut = factory.make_sut(sut_definition)

    assert isinstance(sut, NvidiaNIMApiClient)
    assert sut.uid == "foo/bar:nvidia-nim"
    assert sut.model == "foo/bar"


def test_make_sut_bad_model(factory):
    sut_definition = SUTDefinition(model="bogus", maker="fake", driver="nvidia-nim")
    factory._client = MagicMock()
    factory._client.models.retrieve.side_effect = Exception()
    with pytest.raises(ModelNotSupportedError):
        factory.make_sut(sut_definition)
