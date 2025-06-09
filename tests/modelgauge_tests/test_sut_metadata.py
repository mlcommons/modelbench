import random

import pytest

from modelgauge.dynamic_sut_metadata import _is_date, DynamicSUTMetadata

sut_uids = (
    "amazon-nova-1.0-lite",
    "amazon-nova-1.0-micro",
    "amazon-nova-1.0-pro",
    "athene-v2-chat",
    "aya-expanse-8b",
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
    "cohere-c4ai-command-a-03-2025",
    "deepseek-R1",
    "deepseek-v3",
    "demo_always_angry",
    "demo_always_sorry",
    "demo_random_words",
    "demo_yes_no",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-flash-8b-safety_block_most",
    "gemini-1.5-flash-8b-safety_block_none",
    "gemini-1.5-flash-safety_block_most",
    "gemini-1.5-flash-safety_block_none",
    "gemini-1.5-pro",
    "gemini-1.5-pro-safety_block_most",
    "gemini-1.5-pro-safety_block_none",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-safety_block_most",
    "gemini-2.0-flash-lite-safety_block_none",
    "gemini-2.0-flash-safety_block_most",
    "gemini-2.0-flash-safety_block_none",
    "gemma-2-9b-it",
    "google-gemma-3-27b-it-hf-nebius",
    "gpt-3.5-turbo",
    "gpt-4o",
    "gpt-4o-mini",
    "llama-3-1-tulu-3-70b",
    "llama-3-1-tulu-3-8b",
    "llama-3-70b-chat",
    "llama-3-70b-chat",
    "llama-3.1-405b-instruct-turbo",
    "llama-3.1-8b-instruct-turbo",
    "llama-3.3-70b-instruct-turbo",
    "meta-llama-3_1-8b-instruct-hf-nebius",
    "Mistral-7B-Instruct-v0.2",
    "mistral-8x22b-instruct",
    "mistral-nemo-instruct-2407",
    "mistralai-ministral-8b-2410",
    "mistralai-ministral-8b-2410-moderated",
    "mistralai-mistral-large-2411",
    "mistralai-mistral-large-2411-moderated",
    "Mixtral-8x7B-Instruct-v0.1",
    "nvidia-llama-3-1-nemotron-nano-8b-v1",
    "nvidia-llama-3.1-nemotron-70b-instruct",
    "nvidia-llama-3.3-49b-nemotron-super",
    "nvidia-mistral-nemo-minitron-8b-8k-instruct",
    "nvidia-nemotron-4-340b-instruct",
    "nvidia-nemotron-mini-4b-instruct",
    "olmo-2-0325-32b-instruct",
    "olmo-2-1124-7b-instruct",
    "olmo-7b-0724-instruct",
    "phi-3.5-mini-instruct",
    "phi-3.5-moe-instruct",
    "phi-4",
    "qwen2-5-7b-instruct",
    "qwen2.5-7B-instruct-turbo",
    "qwq-32b",
    "vertexai-mistral-large-2411",
    "yi-1-5-34b-chat",
)


@pytest.mark.parametrize("model", sut_uids)
def test_good_models(model):
    assert DynamicSUTMetadata(model=model, vendor="meta", provider="hf")


@pytest.mark.parametrize("model", sut_uids)
def test_bad_models(model):
    bad = list(r'!@#$%^&*():<?,+=`|\]{}["\'')
    head = random.choice(bad)
    tail = random.choice(bad)
    model = f"{head}{model}{tail}"
    with pytest.raises(ValueError):
        _ = DynamicSUTMetadata(model=model, vendor="vendor", provider="provider", driver="driver", date="20250101")


def test_is_proxied():
    s = DynamicSUTMetadata(model="phi-4", vendor="azure", provider="hf", driver="", date="")
    assert not s.is_proxied()

    s = DynamicSUTMetadata(model="gemma-2-9b-it", vendor="google", provider="cohere", driver="hfproxy", date="20250101")
    assert s.is_proxied()


def test_external_model_name():
    s = DynamicSUTMetadata(model="phi-4", vendor="azure", provider="hf", driver="", date="")
    assert s.external_model_name() == "azure/phi-4"

    s = DynamicSUTMetadata(model="qwen2-5-7b-instruct", vendor="", provider="", driver="", date="")
    assert s.external_model_name() == "qwen2-5-7b-instruct"


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
    assert DynamicSUTMetadata.parse_sut_uid(uid) == DynamicSUTMetadata(
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
    s = DynamicSUTMetadata(model=model, vendor=vendor, provider=provider, driver=driver, date=date)
    assert DynamicSUTMetadata.make_sut_uid(s) == uid


def test__is_date():
    assert _is_date("20200101")
    assert _is_date("2020-01-01")
    assert not _is_date("1234567")
    assert not _is_date("123456789")
    assert not _is_date("not a date")
