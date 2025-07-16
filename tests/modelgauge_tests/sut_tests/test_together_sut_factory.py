import os

from unittest.mock import patch

import pytest

from modelgauge.dynamic_sut_factory import ModelNotSupportedError
from modelgauge.dynamic_sut_metadata import DynamicSUTMetadata
from modelgauge.suts.together_sut_factory import TogetherSUTFactory


def test_make_sut():
    with patch("modelgauge.suts.together_sut_factory.TogetherSUTFactory._find", return_value="google/gemma:together"):
        sut_metadata = DynamicSUTMetadata(model="gemma", maker="google", driver="together")
        found_sut = TogetherSUTFactory.make_sut(sut_metadata)
        assert found_sut is not None
        assert found_sut[1] == "google/gemma:together"


def test_make_sut_bad_model():
    os.environ["TOGETHER_API_KEY"] = "fake-key"

    sut_metadata = DynamicSUTMetadata(model="bogus", maker="fake", driver="together")
    with patch("modelgauge.suts.together_sut_factory.TogetherSUTFactory._find", side_effect=ModelNotSupportedError()):
        with pytest.raises(ModelNotSupportedError):
            _ = TogetherSUTFactory.make_sut(sut_metadata)


def test_find():
    os.environ["TOGETHER_API_KEY"] = "fake-key"

    with patch(
        "modelgauge.suts.together_sut_factory.together.Models.list",
        return_value=[{"id": "google/gemma"}],
    ):
        sut_metadata = DynamicSUTMetadata(model="gemma", maker="google", driver="together")
        assert TogetherSUTFactory._find(sut_metadata) == sut_metadata.external_model_name()


def test_find_bad_model():
    sut_metadata = DynamicSUTMetadata(model="any", maker="any", driver="together")
    with patch(
        "modelgauge.suts.together_sut_factory.together.Models.list",
        return_value=None,
    ):
        with pytest.raises(ModelNotSupportedError):
            _ = TogetherSUTFactory._find(sut_metadata)
