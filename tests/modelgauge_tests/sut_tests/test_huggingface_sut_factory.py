from contextlib import nullcontext
from threading import BoundedSemaphore
from unittest.mock import patch

import pytest

from modelgauge.config import load_secrets_from_config
from modelgauge.dynamic_sut_factory import ProviderNotFoundError
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.huggingface_chat_completion import (
    HuggingFaceChatCompletionDedicatedSUT,
    HuggingFaceChatCompletionServerlessSUT,
)
from modelgauge.suts.huggingface_sut_factory import (
    FEATHERLESS_MAX_IN_FLIGHT,
    HuggingFaceChatCompletionDedicatedSUTFactory,
    HuggingFaceChatCompletionServerlessSUTFactory,
)
from modelgauge_tests.utilities import expensive_tests, FakeObject

RAW_SECRETS = {"hugging_face": {"token": "value"}}


@pytest.fixture
def serverless_factory(monkeypatch):
    monkeypatch.setattr(HuggingFaceChatCompletionServerlessSUTFactory, "_find", lambda *args, **kwargs: "cohere")
    factory = HuggingFaceChatCompletionServerlessSUTFactory(RAW_SECRETS)
    return factory


def test_serverless_make_sut_proxied(serverless_factory):
    sut_definition = SUTDefinition(model="gemma", maker="google", driver="hfrelay", provider="cohere")
    sut = serverless_factory.make_sut(sut_definition)
    assert isinstance(sut, HuggingFaceChatCompletionServerlessSUT)
    assert sut.uid == "google/gemma:cohere:hfrelay"
    assert sut.model == "google/gemma"
    assert sut.provider == "cohere"


def test_serverless_make_sut_direct(serverless_factory):
    sut_definition = SUTDefinition(model="gemma", maker="google", driver="hf-serverless")
    sut = serverless_factory.make_sut(sut_definition)
    assert isinstance(sut, HuggingFaceChatCompletionServerlessSUT)
    assert sut.uid == "google/gemma:hf-serverless"
    assert sut.model == "google/gemma"
    assert sut.provider == "cohere"


def test_serverless_make_sut_does_not_limit_in_flight_for_other_providers(serverless_factory):
    sut_definition = SUTDefinition(model="gemma", maker="google", driver="hf-serverless", provider="cohere")
    sut = serverless_factory.make_sut(sut_definition)
    assert isinstance(sut._in_flight_limit, nullcontext)


def test_serverless_make_sut_limits_in_flight_for_featherless(monkeypatch):
    monkeypatch.setattr(
        HuggingFaceChatCompletionServerlessSUTFactory, "_find", lambda *args, **kwargs: "featherless-ai"
    )
    factory = HuggingFaceChatCompletionServerlessSUTFactory(RAW_SECRETS)
    sut_definition = SUTDefinition(
        model="dolphin-mistral-24b-venice-edition", maker="dphn", driver="hf-serverless", provider="featherless-ai"
    )
    sut = factory.make_sut(sut_definition)
    assert isinstance(sut._in_flight_limit, BoundedSemaphore)
    for _ in range(FEATHERLESS_MAX_IN_FLIGHT):
        assert sut._in_flight_limit.acquire(blocking=False)
    assert not sut._in_flight_limit.acquire(blocking=False)
    for _ in range(FEATHERLESS_MAX_IN_FLIGHT):
        sut._in_flight_limit.release()


def test_serverless_make_sut_no_provider_found():
    with patch(
        "modelgauge.suts.huggingface_sut_factory.HuggingFaceChatCompletionServerlessSUTFactory.find_inference_provider_for",
        return_value=[],
    ):
        factory = HuggingFaceChatCompletionServerlessSUTFactory(RAW_SECRETS)
        with pytest.raises(ProviderNotFoundError):
            factory.make_sut(SUTDefinition.parse("google/gemma:bogus:hfrelay"))


@pytest.fixture
def dedicated_factory(monkeypatch):
    monkeypatch.setattr(
        HuggingFaceChatCompletionDedicatedSUTFactory,
        "_find",
        lambda _, *args, **kwargs: ("endpoint_name", "model_name"),
    )
    factory = HuggingFaceChatCompletionDedicatedSUTFactory(RAW_SECRETS)
    return factory


def test_dedicated_make_sut(dedicated_factory):
    sut_definition = SUTDefinition(model="gemma", maker="google", driver="hf-dedicated")
    sut = dedicated_factory.make_sut(sut_definition)
    assert isinstance(sut, HuggingFaceChatCompletionDedicatedSUT)
    assert sut.uid == "google/gemma:hf-dedicated"
    assert sut.inference_endpoint == "endpoint_name"


def test_dedicated_make_sut_no_endpoint_found():
    with patch(
        "modelgauge.suts.huggingface_sut_factory.hfh.list_inference_endpoints",
        return_value=[],
    ):
        factory = HuggingFaceChatCompletionDedicatedSUTFactory(RAW_SECRETS)
        with pytest.raises(ProviderNotFoundError):
            factory.make_sut(SUTDefinition.parse("google/gemma:hf-dedicated"))


def test_dedicated_list_suts():
    with patch(
        "modelgauge.suts.huggingface_sut_factory.HuggingFaceChatCompletionDedicatedSUTFactory.get_endpoints",
        return_value=[FakeObject(repository="google/gemma")],
    ):
        factory = HuggingFaceChatCompletionDedicatedSUTFactory(RAW_SECRETS)

        assert "google/gemma:hf-dedicated" in [s.uid for s in factory.list_suts()]


@expensive_tests
def test_connection():
    factory = HuggingFaceChatCompletionServerlessSUTFactory(load_secrets_from_config(path="."))
    sut_definition = SUTDefinition(model="gemma-3-27b-it", maker="google", driver="hfrelay", provider="nebius")
    sut = factory.make_sut(sut_definition)
    assert isinstance(sut, HuggingFaceChatCompletionServerlessSUT)
    assert sut.uid == "google/gemma-3-27b-it:nebius:hfrelay"
    assert sut.model == "google/gemma-3-27b-it"
    assert sut.provider == "nebius"
