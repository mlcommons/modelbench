from unittest.mock import patch

import pytest
from modelgauge import dynamic_sut_finder
from modelgauge.dynamic_sut_factory import UnknownProxyError
from modelgauge.suts.huggingface_chat_completion import HuggingFaceChatCompletionServerlessSUT


def test_make_dynamic_sut():
    with pytest.raises(UnknownProxyError):
        _ = dynamic_sut_finder.make_dynamic_sut_for("google:gemma:nebius:bogusproxy:20250101")

    with patch(
        "modelgauge.suts.huggingface_sut_factory.HuggingFaceChatCompletionServerlessSUTFactory.find",
        return_value="cohere",
    ):
        registrable_sut = dynamic_sut_finder.make_dynamic_sut_for("google:gemma:cohere:hfrelay")
        assert registrable_sut[0:4] == (
            HuggingFaceChatCompletionServerlessSUT,
            "google:gemma:cohere:hfrelay",
            "google/gemma",
            "cohere",
        )
