import pytest
from modelgauge.dynamic_sut_maker import DynamicSUTMaker


def test_parse_sut_name():
    assert DynamicSUTMaker.parse_sut_name("hf/nebius/google/gemma") == ("hf", "nebius", "google", "gemma")
    assert DynamicSUTMaker.parse_sut_name("nebius/google/gemma") == ("", "nebius", "google", "gemma")
    assert DynamicSUTMaker.parse_sut_name("google/gemma") == ("", "", "google", "gemma")
    assert DynamicSUTMaker.parse_sut_name("gemma") == ("", "", "", "gemma")
    with pytest.raises(ValueError):
        _ = DynamicSUTMaker.parse_sut_name("extra/hf/nebius/google/gemma")
    with pytest.raises(ValueError):
        _ = DynamicSUTMaker.parse_sut_name("hf/nebius/google/gemma/extra")
    with pytest.raises(ValueError):
        _ = DynamicSUTMaker.parse_sut_name("hf/nebius/google/")


def extract_model_name():
    assert DynamicSUTMaker.extract_model_name("hf/nebius/google/gemma") == "google/gemma"
    assert DynamicSUTMaker.extract_model_name("google/gemma") == "google/gemma"
    assert DynamicSUTMaker.extract_model_name("gemma") == "gemma"
    with pytest.raises(ValueError):
        _ = DynamicSUTMaker.extract_model_name("hf/nebius/google/gemma/extra")
    with pytest.raises(ValueError):
        _ = DynamicSUTMaker.extract_model_name("extra/hf/nebius/google/gemma")
    with pytest.raises(ValueError):
        _ = DynamicSUTMaker.extract_model_name("hf/nebius/google/")
