from unittest.mock import patch

from plugins.huggingface.modelgauge.suts.huggingface_sut_maker import (
    HuggingFaceSUTMaker,
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
        assert HuggingFaceSUTMaker.make_sut(model_name="hf/cohere/google/gemma") is not None

    with patch(
        "plugins.huggingface.modelgauge.suts.huggingface_sut_maker.HuggingFaceChatCompletionServerlessSUTMaker.find",
        return_value=None,
    ):
        assert HuggingFaceSUTMaker.make_sut(model_name="hf/cohere/google/gemma") is None
