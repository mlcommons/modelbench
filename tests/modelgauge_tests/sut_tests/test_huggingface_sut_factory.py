from unittest.mock import patch

import pytest

from modelgauge.dynamic_sut_factory import ProviderNotFoundError
from modelgauge.dynamic_sut_metadata import DynamicSUTMetadata, UnknownSUTDriverError
from modelgauge.suts.huggingface_sut_factory import make_sut


def test_make_sut():
    with patch(
        "modelgauge.suts.huggingface_sut_factory.HuggingFaceChatCompletionServerlessSUTFactory._find",
        return_value="cohere",
    ):
        sut_metadata = DynamicSUTMetadata(model="gemma", maker="google", driver="hfrelay", provider="cohere")
        assert make_sut(sut_metadata) is not None


def test_make_sut_bad_proxy():
    with pytest.raises(UnknownSUTDriverError):
        _ = make_sut(DynamicSUTMetadata.parse_sut_uid("google/gemma:cohere:bogus"))


def test_make_sut_bad_provider():
    with patch(
        "modelgauge.suts.huggingface_sut_factory.find_inference_provider_for",
        return_value={"example": ""},
    ):
        with pytest.raises(ProviderNotFoundError):
            _ = make_sut(DynamicSUTMetadata.parse_sut_uid("google/gemma:bogus:hfrelay"))
