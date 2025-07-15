import pytest
from unittest.mock import patch

from modelgauge.dynamic_sut_factory import ModelNotSupportedError
from modelgauge.dynamic_sut_metadata import DynamicSUTMetadata
from modelgauge.suts.openai_client import OpenAIApiKey, OpenAIChat
from modelgauge.suts.openai_sut_factory import OpenAISUTFactory


def test_make_sut():
    with patch(
        "modelgauge.suts.openai_sut_factory.OpenAISUTFactory._model_exists",
        return_value=True,
    ):
        sut_metadata = DynamicSUTMetadata(model="gpt-4o", maker="openai", driver="openai")
        factory = OpenAISUTFactory()
        sut = factory.make_sut(sut_metadata)
    assert sut[0] == OpenAIChat
    assert sut[1] == "openai/gpt-4o:openai"
    assert sut[2] == "gpt-4o"


def test_make_sut_no_maker():
    with patch(
        "modelgauge.suts.openai_sut_factory.OpenAISUTFactory._model_exists",
        return_value=True,
    ):
        sut_metadata = DynamicSUTMetadata(model="gpt-4o", driver="openai")
        factory = OpenAISUTFactory()
        sut = factory.make_sut(sut_metadata)
    assert sut[0] == OpenAIChat
    assert sut[1] == "gpt-4o:openai"
    assert sut[2] == "gpt-4o"


def test_make_unknown_sut_raises_error():
    with patch(
        "modelgauge.suts.openai_sut_factory.OpenAISUTFactory._model_exists",
        return_value=False,
    ):
        sut_metadata = DynamicSUTMetadata(model="gpt-4o", maker="openai", driver="openai")
        factory = OpenAISUTFactory()
        with pytest.raises(ModelNotSupportedError):
            factory.make_sut(sut_metadata)
