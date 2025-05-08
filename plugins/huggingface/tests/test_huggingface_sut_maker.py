from unittest.mock import patch

import huggingface_hub as hfh

import pytest

from modelgauge.dynamic_sut_maker import (
    ModelNotSupportedError,
    ProviderNotFoundError,
    UnknownProxyError,
)

from plugins.huggingface.modelgauge.suts.huggingface_sut_maker import (
    HuggingFaceSUTMaker,
    make_sut,
)


def test_make_uid():
    assert (
        HuggingFaceSUTMaker.make_sut_id("CohereLabs/c4ai-command-a-03-2025") == "coherelabs-c4ai-command-a-03-2025-hf"
    )


def test_make_sut_id():
    sut_id = HuggingFaceSUTMaker.make_sut_id("hf/nebius/google/gemma-7b-it")
    assert sut_id == "google-gemma-7b-it-hf-nebius"

    sut_id = HuggingFaceSUTMaker.make_sut_id("hf/google/gemma-7b-it")
    assert sut_id == "google-gemma-7b-it-hf"

    sut_id = HuggingFaceSUTMaker.make_sut_id("google/gemma-7b-it")
    assert sut_id == "google-gemma-7b-it-hf"


def test_make_sut():
    with patch(
        "plugins.huggingface.modelgauge.suts.huggingface_sut_maker.HuggingFaceChatCompletionServerlessSUTMaker.find",
        return_value="cohere",
    ):
        assert make_sut(model_name="hf/cohere/google/gemma") is not None

    with patch(
        "plugins.huggingface.modelgauge.suts.huggingface_sut_maker.HuggingFaceChatCompletionServerlessSUTMaker.find",
        return_value=None,
    ):
        assert make_sut(model_name="hf/cohere/google/gemma") is None


def test_make_sut_errors():
    with pytest.raises(UnknownProxyError):
        make_sut("bogus/cohere/google/gemma")

    with patch(
        "plugins.huggingface.modelgauge.suts.huggingface_sut_maker.find_inference_provider_for",
        return_value={"example": ""},
    ):
        with pytest.raises(ProviderNotFoundError):
            _ = make_sut(model_name="hf/cohere/google/gemma")

    with patch(
        "plugins.huggingface.modelgauge.suts.huggingface_sut_maker.hfh.model_info",
        side_effect=hfh.errors.RepositoryNotFoundError("error"),
    ):
        with pytest.raises(ModelNotSupportedError):
            _ = make_sut(model_name="hf/cohere/google/fake")
