import pytest
from modelgauge.dynamic_sut_maker import _is_date, DynamicSUTMaker
from modelgauge.sut_metadata import SUTMetadata


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


@pytest.mark.parametrize(
    "uid,vendor,model,provider,driver,date",
    (
        ("gemma-3-27b-it", "", "gemma-3-27b-it", "", "", ""),
        ("google:gemma-3-27b-it", "google", "gemma-3-27b-it", "", "", ""),
        ("google:gemma-3-27b-it:nebius:hfrelay", "google", "gemma-3-27b-it", "nebius", "hfrelay", ""),
        ("google:gemma-3-27b-it:nebius:hfrelay:20250507", "google", "gemma-3-27b-it", "nebius", "hfrelay", "20250507"),
    ),
)
def test_make_sut_uid(uid, vendor, model, provider, driver, date):
    s = SUTMetadata(model=model, vendor=vendor, provider=provider, driver=driver, date=date)
    assert DynamicSUTMaker.make_sut_uid(s) == uid


def test__is_date():
    assert _is_date("20200101")
    assert _is_date("2020-01-01")
    assert not _is_date("1234567")
    assert not _is_date("123456789")
    assert not _is_date("not a date")
