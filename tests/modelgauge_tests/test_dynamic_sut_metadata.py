import pytest

from modelgauge.dynamic_sut_metadata import _is_date, DynamicSUTMetadata, MissingModelError, UnknownSUTDriverError


def test_external_model_name():
    s = DynamicSUTMetadata(model="phi-4", maker="azure", provider="", driver="together", date="")
    assert s.external_model_name() == "azure/phi-4"

    s = DynamicSUTMetadata(model="qwen2-5-7b-instruct", maker="", provider="", driver="huggingface", date="")
    assert s.external_model_name() == "qwen2-5-7b-instruct"


@pytest.mark.parametrize(
    "uid,maker,model,provider,driver,date",
    (
        ("google/gemma-3-27b-it:nebius:hfrelay:20250507", "google", "gemma-3-27b-it", "nebius", "hfrelay", "20250507"),
        (
            "google/gemini-1.5-flash-8b-safety_block_most:vertexai:20250507",
            "google",
            "gemini-1.5-flash-8b-safety_block_most",
            "",
            "vertexai",
            "20250507",
        ),
        ("meta/llama-3.1-405b-instruct-turbo:llama", "meta", "llama-3.1-405b-instruct-turbo", "", "llama", ""),
        ("phi-3.5-moe-instruct:azure", "", "phi-3.5-moe-instruct", "", "azure", ""),
    ),
)
def test_parse_sut_uid(uid, maker, model, provider, driver, date):
    if not model:
        with pytest.raises(MissingModelError):
            _ = DynamicSUTMetadata.parse_sut_uid(uid)
    if not driver:
        with pytest.raises(UnknownSUTDriverError):
            _ = DynamicSUTMetadata.parse_sut_uid(uid)
    if model and driver:
        assert DynamicSUTMetadata.parse_sut_uid(uid) == DynamicSUTMetadata(
            model=model, maker=maker, provider=provider, driver=driver, date=date
        )


@pytest.mark.parametrize(
    "legacy_uid,model,driver",
    (
        ("gemma-2-9b-it-simpo-hf", "gemma-2-9b-it-simpo-hf", "huggingface"),
        ("llama-3.1-405b-instruct-turbo-together", "llama-3.1-405b-instruct-turbo-together", "together"),
    ),
)
def parse_legacy_dut_uids(legacy_uid, model, driver):
    metadata = DynamicSUTMetadata.parse_sut_uid(legacy_uid)
    assert metadata.model == model
    assert metadata.driver == driver


@pytest.mark.parametrize(
    "uid,maker,model,provider,driver,date",
    (
        ("gemma-3-27b-it:vertexai", "", "gemma-3-27b-it", "", "vertexai", ""),
        ("google/gemma-3-27b-it:vertexai", "google", "gemma-3-27b-it", "", "vertexai", ""),
        ("google/gemma-3-27b-it:nebius:hfrelay", "google", "gemma-3-27b-it", "nebius", "hfrelay", ""),
        ("google/gemma-3-27b-it:nebius:hfrelay:20250507", "google", "gemma-3-27b-it", "nebius", "hfrelay", "20250507"),
        ("bogus", "", "bogus", "", "", ""),
    ),
)
def test_make_sut_uid(uid, maker, model, provider, driver, date):
    if not driver:
        with pytest.raises(UnknownSUTDriverError):
            s = DynamicSUTMetadata(model=model, maker=maker, provider=provider, driver=driver, date=date)
            _ = DynamicSUTMetadata.make_sut_uid(s)
    else:
        metadata = DynamicSUTMetadata(model=model, maker=maker, provider=provider, driver=driver, date=date)
        assert DynamicSUTMetadata.make_sut_uid(metadata) == uid


def test__is_date():
    assert _is_date("20200101")
    assert _is_date("2020-01-01")
    assert not _is_date("1234567")
    assert not _is_date("123456789")
    assert not _is_date("not a date")


def test_str():
    metadata = DynamicSUTMetadata(model="superhero", maker="marvel", provider="", driver="together", date="")
    assert str(metadata) == "marvel/superhero:together"

    metadata = DynamicSUTMetadata(model="super", maker="", provider="nebius", driver="hfrelay", date="20200101")
    assert str(metadata) == "super:nebius:hfrelay:20200101"
