import pytest
from modelgauge.dynamic_sut_maker import DynamicSUTMaker


def test_parse_sut_name():
    with pytest.raises(ValueError):
        _ = DynamicSUTMaker.parse_sut_name("/a/b/c/d/e/f/g")

    assert DynamicSUTMaker.parse_sut_name("hf/nebius/google/gemma-7b-it") == ("hf", "nebius", "google", "gemma-7b-it")
    assert DynamicSUTMaker.parse_sut_name("hf/google/gemma-7b-it") == ("", "hf", "google", "gemma-7b-it")
    assert DynamicSUTMaker.parse_sut_name("meta-llama/Llama-3.1-8B-Instruct") == (
        "",
        "",
        "meta-llama",
        "Llama-3.1-8B-Instruct",
    )


def test_extract_model_name():
    assert DynamicSUTMaker.extract_model_name("hf/nebius/google/gemma-7b-it") == "google/gemma-7b-it"
    assert DynamicSUTMaker.extract_model_name("hf/google/gemma-7b-it") == "google/gemma-7b-it"
    assert DynamicSUTMaker.extract_model_name("google/gemma-7b-it") == "google/gemma-7b-it"
    assert DynamicSUTMaker.extract_model_name("gemma-7b-it") == "gemma-7b-it"
