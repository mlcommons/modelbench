import pytest
from unittest.mock import patch

from modelgauge.dynamic_sut_factory import ModelNotSupportedError
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.meta_llama_client import MetaLlamaSUT
from modelgauge.suts.meta_llama_factory import LlamaSUTFactory


@pytest.fixture
def factory():
    return LlamaSUTFactory({"meta_llama": {"api_key": "value"}})


def test_make_sut(factory):
    with patch("modelgauge.suts.meta_llama_factory.LlamaSUTFactory._get_model_name", return_value="Foo/Bar"):
        sut_definition = SUTDefinition(model="bar", maker="foo", driver="llama")
        sut = factory.make_sut(sut_definition)

        assert isinstance(sut, MetaLlamaSUT)
        assert sut.uid == "foo/bar:llama"
        assert sut.model == "Foo/Bar"


def test_make_sut_bad_model(factory):
    sut_definition = SUTDefinition(model="bogus", maker="fake", driver="llama")
    with patch("modelgauge.suts.meta_llama_factory.LlamaSUTFactory._get_model_name", return_value=None):
        with pytest.raises(ModelNotSupportedError):
            factory.make_sut(sut_definition)
