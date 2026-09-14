from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modelgauge.dynamic_sut_factory import ModelNotSupportedError
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.openai_client import OpenAIResponsesRequest
from modelgauge.suts.openrouter_sut_factory import (
    OPENROUTER_BASE_URL,
    OpenRouterResponsesSUT,
    OpenRouterSUTFactory,
)


@pytest.fixture
def factory() -> OpenRouterSUTFactory:
    result = OpenRouterSUTFactory(raw_secrets={"openrouter": {"api_key": "test-key"}})
    result._client = MagicMock()
    return result


def set_models(factory: OpenRouterSUTFactory, *model_ids: str) -> None:
    factory._client.models.list.return_value = [SimpleNamespace(id=model_id) for model_id in model_ids]


def test_openrouter_client_uses_expected_base_url() -> None:
    factory = OpenRouterSUTFactory(raw_secrets={"openrouter": {"api_key": "test-key"}})

    assert str(factory.client.base_url).rstrip("/") == OPENROUTER_BASE_URL


def test_list_suts_preserves_canonical_model_identity(
    factory: OpenRouterSUTFactory,
) -> None:
    set_models(
        factory,
        "openai/gpt-5.6-sol",
        "google/gemini-3.8-flash",
    )

    assert [sut.uid for sut in factory.list_suts()] == [
        "openai/gpt-5.6-sol:openrouter",
        "google/gemini-3.8-flash:openrouter",
    ]


def test_list_suts_skips_colon_variants_until_uid_encoding_is_defined(
    factory: OpenRouterSUTFactory,
) -> None:
    set_models(
        factory,
        "openai/gpt-5.6-sol",
        "openai/gpt-5.6-sol:floor",
        "meta-llama/llama-3.3-70b-instruct:nitro",
    )

    assert [sut.uid for sut in factory.list_suts()] == ["openai/gpt-5.6-sol:openrouter"]


def test_make_sut_reuses_openai_responses_path(
    factory: OpenRouterSUTFactory,
) -> None:
    set_models(factory, "openai/gpt-5.6-sol")

    sut = factory.make_sut(
        SUTDefinition(
            maker="openai",
            model="gpt-5.6-sol",
            driver="openrouter",
        )
    )

    assert isinstance(sut, OpenRouterResponsesSUT)
    assert sut.uid == "openai/gpt-5.6-sol:openrouter"
    assert sut.model == "openai/gpt-5.6-sol"
    assert sut.provider is None


def test_make_sut_rejects_unknown_model(
    factory: OpenRouterSUTFactory,
) -> None:
    set_models(factory, "openai/gpt-5.6-sol")

    with pytest.raises(ModelNotSupportedError, match="not found or not available"):
        factory.make_sut(
            SUTDefinition(
                maker="openai",
                model="does-not-exist",
                driver="openrouter",
            )
        )


def test_provider_component_pins_openrouter_routing(
    factory: OpenRouterSUTFactory,
) -> None:
    set_models(factory, "openai/gpt-5.6-sol")
    sut = factory.make_sut(
        SUTDefinition(
            maker="openai",
            model="gpt-5.6-sol",
            provider="openai",
            driver="openrouter",
        )
    )
    request = OpenAIResponsesRequest(
        input=[],
        model="openai/gpt-5.6-sol",
    )
    expected_response = MagicMock()
    factory._client.responses.create.return_value = expected_response

    response = sut._call_client(request)

    assert response is expected_response
    factory._client.responses.create.assert_called_once()
    kwargs = factory._client.responses.create.call_args.kwargs
    assert kwargs["extra_body"] == {
        "provider": {
            "only": ["openai"],
            "allow_fallbacks": False,
        }
    }


def test_unpinned_openrouter_routing_keeps_default_behavior(
    factory: OpenRouterSUTFactory,
) -> None:
    set_models(factory, "openai/gpt-5.6-sol")
    sut = factory.make_sut(
        SUTDefinition(
            maker="openai",
            model="gpt-5.6-sol",
            driver="openrouter",
        )
    )
    request = OpenAIResponsesRequest(
        input=[],
        model="openai/gpt-5.6-sol",
    )

    sut._call_client(request)

    kwargs = factory._client.responses.create.call_args.kwargs
    assert "extra_body" not in kwargs
