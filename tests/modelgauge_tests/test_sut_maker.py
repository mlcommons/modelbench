import pytest
from modelgauge.sut import SUTMaker


def test_parse_sut_name():
    with pytest.raises(ValueError):
        _ = SUTMaker.parse_sut_name("/a/b/c/d/e/f/g")

    assert SUTMaker.parse_sut_name("hf/nebius/google/gemma-7b-it") == ("hf", "nebius", "google", "gemma-7b-it")
    assert SUTMaker.parse_sut_name("hf/google/gemma-7b-it") == (None, "hf", "google", "gemma-7b-it")
    assert SUTMaker.parse_sut_name("meta-llama/Llama-3.1-8B-Instruct") == (
        None,
        None,
        "meta-llama",
        "Llama-3.1-8B-Instruct",
    )


def test_extract_model_name():
    assert SUTMaker.extract_model_name("hf/nebius/google/gemma-7b-it") == "google/gemma-7b-it"
    assert SUTMaker.extract_model_name("hf/google/gemma-7b-it") == "google/gemma-7b-it"
    assert SUTMaker.extract_model_name("google/gemma-7b-it") == "google/gemma-7b-it"
    assert SUTMaker.extract_model_name("gemma-7b-it") == "gemma-7b-it"
