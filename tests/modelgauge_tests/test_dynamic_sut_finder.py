from unittest.mock import patch

import pytest
from modelgauge import dynamic_sut_finder
from modelgauge.dynamic_sut_maker import UnknownProxyError
from modelgauge.suts.huggingface_chat_completion import HuggingFaceChatCompletionServerlessSUT


def test_make_dynamic_sut():
    with pytest.raises(UnknownProxyError):
        _ = dynamic_sut_finder.make_dynamic_sut_for("bogus/sut/name")

    with patch(
        "modelgauge.suts.huggingface_sut_maker.HuggingFaceChatCompletionServerlessSUTMaker.find",
        return_value="cohere",
    ):
        registrable_sut = dynamic_sut_finder.make_dynamic_sut_for("hf/cohere/google/gemma")
        assert registrable_sut[0:4] == (
            HuggingFaceChatCompletionServerlessSUT,
            "google-gemma-hf-cohere",
            "google/gemma",
            "cohere",
        )
