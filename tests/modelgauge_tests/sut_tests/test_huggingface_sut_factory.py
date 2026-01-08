from unittest.mock import patch

import pytest

from modelgauge.config import load_secrets_from_config
from modelgauge.dynamic_sut_factory import ModelNotSupportedError, ProviderNotFoundError
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.huggingface_chat_completion import (
    HuggingFaceChatCompletionDedicatedSUT,
    HuggingFaceChatCompletionServerlessSUT,
)
from modelgauge.suts.huggingface_sut_factory import (
    HuggingFaceChatCompletionDedicatedSUTFactory,
    HuggingFaceChatCompletionServerlessSUTFactory,
    HuggingFaceSUTFactory,
)
from modelgauge_tests.utilities import expensive_tests

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
    sut_definition = SUTDefinition(model="gemma", maker="google", driver="hf")
    sut = serverless_factory.make_sut(sut_definition)
    assert isinstance(sut, HuggingFaceChatCompletionServerlessSUT)
    assert sut.uid == "google/gemma:hf"
    assert sut.model == "google/gemma"
    assert sut.provider == "cohere"


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
    sut_definition = SUTDefinition(model="gemma", maker="google", driver="hf", provider="cohere")
    sut = dedicated_factory.make_sut(sut_definition)
    assert isinstance(sut, HuggingFaceChatCompletionDedicatedSUT)
    assert sut.uid == "google/gemma:cohere:hf"
    assert sut.inference_endpoint == "endpoint_name"


def test_dedicated_make_sut_no_provider(dedicated_factory):
    sut_definition = SUTDefinition(model="gemma", maker="google", driver="hf")
    sut = dedicated_factory.make_sut(sut_definition)
    assert isinstance(sut, HuggingFaceChatCompletionDedicatedSUT)
    assert sut.uid == "google/gemma:hf"
    assert sut.inference_endpoint == "endpoint_name"


def test_dedicated_make_sut_no_endpoint_found():
    with patch(
        "modelgauge.suts.huggingface_sut_factory.hfh.list_inference_endpoints",
        return_value=[],
    ):
        factory = HuggingFaceChatCompletionDedicatedSUTFactory(RAW_SECRETS)
        with pytest.raises(ProviderNotFoundError):
            factory.make_sut(SUTDefinition.parse("google/gemma:hf"))


@pytest.fixture
def super_factory(serverless_factory, dedicated_factory):
    factory = HuggingFaceSUTFactory(RAW_SECRETS)
    factory.serverless_factory = serverless_factory
    factory.dedicated_factory = dedicated_factory
    return factory


def test_make_sut_proxied(super_factory):

    sut_definition = SUTDefinition(model="gemma", maker="google", driver="hfrelay", provider="cohere")
    sut = super_factory.make_sut(sut_definition)

    assert isinstance(sut, HuggingFaceChatCompletionServerlessSUT)
    assert sut.uid == "google/gemma:cohere:hfrelay"
    assert sut.model == "google/gemma"
    assert sut.provider == "cohere"


def test_make_sut_direct_serverless(super_factory):
    sut_definition = SUTDefinition(model="gemma", maker="google", driver="hf")
    sut = super_factory.make_sut(sut_definition)
    assert isinstance(sut, HuggingFaceChatCompletionServerlessSUT)
    assert sut.uid == "google/gemma:hf"
    assert sut.model == "google/gemma"
    assert sut.provider == "cohere"


def test_make_sut_direct_dedicated(dedicated_factory):
    # Serverless can't find the provider, so try dedicated.
    with patch(
        "modelgauge.suts.huggingface_sut_factory.HuggingFaceChatCompletionServerlessSUTFactory.find_inference_provider_for",
        return_value=[],
    ):
        factory = HuggingFaceSUTFactory(RAW_SECRETS)
        factory.serverless_factory = HuggingFaceChatCompletionServerlessSUTFactory(RAW_SECRETS)
        factory.dedicated_factory = dedicated_factory

        sut_definition = SUTDefinition(model="gemma", maker="google", driver="hf")
        sut = factory.make_sut(sut_definition)
    assert isinstance(sut, HuggingFaceChatCompletionDedicatedSUT)
    assert sut.uid == "google/gemma:hf"
    assert sut.inference_endpoint == "endpoint_name"


def test_make_sut_no_sut_found():
    # Both serverless and dedicated factories can't find anything.
    with patch(
        "modelgauge.suts.huggingface_sut_factory.HuggingFaceChatCompletionServerlessSUTFactory.find_inference_provider_for",
        return_value=[],
    ):
        with patch(
            "modelgauge.suts.huggingface_sut_factory.hfh.list_inference_endpoints",
            return_value=[],
        ):
            factory = HuggingFaceSUTFactory(RAW_SECRETS)
            factory.serverless_factory = HuggingFaceChatCompletionServerlessSUTFactory(RAW_SECRETS)
            factory.dedicated_factory = HuggingFaceChatCompletionDedicatedSUTFactory(RAW_SECRETS)

            with pytest.raises(ModelNotSupportedError):
                factory.make_sut(SUTDefinition.parse("google/gemma:hf"))


@expensive_tests
def test_connection():
    factory = HuggingFaceSUTFactory(load_secrets_from_config(path="."))
    sut_definition = SUTDefinition(model="gemma-3-27b-it", maker="google", driver="hfrelay", provider="nebius")
    sut = factory.make_sut(sut_definition)
    assert isinstance(sut, HuggingFaceChatCompletionServerlessSUT)
    assert sut.uid == "google/gemma-3-27b-it:nebius:hfrelay"
    assert sut.model == "google/gemma-3-27b-it"
    assert sut.provider == "nebius"
