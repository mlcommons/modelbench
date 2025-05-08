import pytest
from modelgauge.dynamic_sut_maker import _is_date, DynamicSUTMaker
from modelgauge.sut import SUTMetadata


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


@pytest.mark.parametrize(
    "uid,vendor,model,provider,driver,date",
    (
        ("google:gemma-3-27b-it:nebius:hfrelay:20250507", "google", "gemma-3-27b-it", "nebius", "hfrelay", "20250507"),
        (
            "google:gemini-1.5-flash-8b-safety_block_most:vertexai:20250507",
            "google",
            "gemini-1.5-flash-8b-safety_block_most",
            "vertexai",
            "",
            "20250507",
        ),
        ("meta:llama-3.1-405b-instruct-turbo", "meta", "llama-3.1-405b-instruct-turbo", "", "", ""),
        ("phi-3.5-moe-instruct", "", "phi-3.5-moe-instruct", "", "", ""),
    ),
)
def test_parse_sut_uid(uid, vendor, model, provider, driver, date):
    assert DynamicSUTMaker.parse_sut_uid(uid) == SUTMetadata(
        model=model, vendor=vendor, provider=provider, driver=driver, date=date
    )


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


def test__is_date():
    assert _is_date("20200101")
    assert _is_date("2020-01-01")
    assert not _is_date("1234567")
    assert not _is_date("123456789")
    assert not _is_date("not a date")
