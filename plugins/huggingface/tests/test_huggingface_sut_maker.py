from unittest.mock import patch

import huggingface_hub as hfh

import pytest

from modelgauge.dynamic_sut_maker import (
    ModelNotSupportedError,
    ProviderNotFoundError,
    UnknownProxyError,
)

from plugins.huggingface.modelgauge.suts.huggingface_sut_maker import make_sut


def test_make_sut():
    with patch(
        "plugins.huggingface.modelgauge.suts.huggingface_sut_maker.HuggingFaceChatCompletionServerlessSUTMaker.find",
        return_value="cohere",
    ):
        assert make_sut(sut_name="google:gemma:cohere:hfrelay") is not None

    with patch(
        "plugins.huggingface.modelgauge.suts.huggingface_sut_maker.HuggingFaceChatCompletionServerlessSUTMaker.find",
        return_value=None,
    ):
        assert make_sut(sut_name="google:gemma:cohere:hfrelay") is None


def test_make_sut_bad_proxy():
    with pytest.raises(UnknownProxyError):
        make_sut("google:gemma:cohere:bogus")


def test_make_sut_bad_provider():
    with patch(
        "plugins.huggingface.modelgauge.suts.huggingface_sut_maker.find_inference_provider_for",
        return_value={"example": ""},
    ):
        with pytest.raises(ProviderNotFoundError):
            _ = make_sut(sut_name="google:gemma:bogus:hfrelay")


def test_make_sut_bad_model():
    with patch(
        "plugins.huggingface.modelgauge.suts.huggingface_sut_maker.hfh.model_info",
        side_effect=hfh.errors.RepositoryNotFoundError("error"),
    ):
        with pytest.raises(ModelNotSupportedError):
            _ = make_sut(sut_name="google:fake:cohere:hfrelay")
