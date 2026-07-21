from unittest.mock import MagicMock

import pytest
from modelgauge_tests.utilities import expensive_tests

from modelgauge.config import load_secrets_from_config
from modelgauge.dynamic_sut_factory import ModelNotSupportedError
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.together_client import TogetherChatSUT, TogetherDedicatedChatSUT
from modelgauge.suts.together_sut_factory import TogetherSUTFactory


@pytest.fixture
def together_factory():
    factory = TogetherSUTFactory({"together": {"api_key": "some-key"}})
    factory.client = MagicMock()
    return factory


def test_find_serverless(together_factory):
    together_factory.client.chat.completions.create.return_value = {}
    result = together_factory._find_serverless("google/gemma")
    assert result == "google/gemma"
    together_factory.client.chat.completions.create.assert_called_once_with(
        model="google/gemma",
        messages=[{"role": "user", "content": "Anybody home?"}],
    )


def test_find_serverless_bad_model(together_factory):
    together_factory.client.chat.completions.create.side_effect = Exception("Model not available")
    result = together_factory._find_serverless("google/gemma")
    assert result is None


def test_find_dedicated(together_factory):
    mock_endpoint = MagicMock()
    mock_endpoint.model = "google/gemma"
    mock_endpoint.name = "my-dedicated-endpoint"
    together_factory.client.endpoints.list.return_value.data = [mock_endpoint]

    result = together_factory._find_dedicated("google/gemma")
    assert result == "my-dedicated-endpoint"
    together_factory.client.endpoints.list.assert_called_once_with(type="dedicated", mine=True)


def test_find_dedicated_not_found(together_factory):
    together_factory.client.endpoints.list.return_value.data = []
    result = together_factory._find_dedicated("google/gemma")
    assert result is None


def test_find_dedicated_error(together_factory):
    together_factory.client.endpoints.list.side_effect = Exception("API error")
    result = together_factory._find_dedicated("google/gemma")
    assert result is None


def test_make_sut_dedicated_preferred_by_default(together_factory):
    together_factory.client.chat.completions.create.return_value = {}
    mock_endpoint = MagicMock()
    mock_endpoint.model = "google/gemma"
    mock_endpoint.name = "my-dedicated-endpoint"
    together_factory.client.endpoints.list.return_value.data = [mock_endpoint]

    sut_definition = SUTDefinition(model="gemma", maker="google", driver="together")
    sut = together_factory.make_sut(sut_definition)
    assert isinstance(sut, TogetherDedicatedChatSUT)
    assert sut.uid == "google/gemma:together"
    assert sut.model == "my-dedicated-endpoint"
    assert sut.api_key == "some-key"


def test_make_sut_falls_back_to_serverless(together_factory):
    together_factory.client.chat.completions.create.return_value = {}
    together_factory.client.endpoints.list.return_value.data = []

    sut_definition = SUTDefinition(model="gemma", maker="google", driver="together")
    sut = together_factory.make_sut(sut_definition)
    assert isinstance(sut, TogetherChatSUT)
    assert sut.uid == "google/gemma:together"
    assert sut.model == "google/gemma"
    assert sut.api_key == "some-key"


def test_make_sut_prefer_dedicated_false_prefers_serverless(together_factory):
    together_factory.prefer_dedicated = False
    together_factory.client.chat.completions.create.return_value = {}
    mock_endpoint = MagicMock()
    mock_endpoint.model = "google/gemma"
    mock_endpoint.name = "my-dedicated-endpoint"
    together_factory.client.endpoints.list.return_value.data = [mock_endpoint]

    sut_definition = SUTDefinition(model="gemma", maker="google", driver="together")
    sut = together_factory.make_sut(sut_definition)
    assert isinstance(sut, TogetherChatSUT)


def test_make_sut_prefer_dedicated_false_falls_back_to_dedicated(together_factory):
    together_factory.prefer_dedicated = False
    together_factory.client.chat.completions.create.side_effect = Exception("not serverless")
    mock_endpoint = MagicMock()
    mock_endpoint.model = "google/gemma"
    mock_endpoint.name = "my-dedicated-endpoint"
    together_factory.client.endpoints.list.return_value.data = [mock_endpoint]

    sut_definition = SUTDefinition(model="gemma", maker="google", driver="together")
    sut = together_factory.make_sut(sut_definition)
    assert isinstance(sut, TogetherDedicatedChatSUT)
    assert sut.model == "my-dedicated-endpoint"


@pytest.mark.parametrize(
    ("env_value", "expected"),
    (
        (None, True),
        ("1", True),
        ("true", True),
        ("True", True),
        ("yes", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("whatever", False),
    ),
)
def test_prefer_dedicated_env_var(monkeypatch, env_value, expected):
    if env_value is None:
        monkeypatch.delenv("TOGETHER_PREFER_DEDICATED", raising=False)
    else:
        monkeypatch.setenv("TOGETHER_PREFER_DEDICATED", env_value)
    factory = TogetherSUTFactory({"together": {"api_key": "some-key"}})
    assert factory.prefer_dedicated is expected


def test_prefer_dedicated_constructor_overrides_env(monkeypatch):
    monkeypatch.setenv("TOGETHER_PREFER_DEDICATED", "false")
    factory = TogetherSUTFactory({"together": {"api_key": "some-key"}}, prefer_dedicated=True)
    assert factory.prefer_dedicated is True


def test_make_sut_prefer_dedicated_param_persists_at_object_level(together_factory):
    together_factory.client.chat.completions.create.return_value = {}
    together_factory.client.endpoints.list.return_value.data = []

    sut_definition = SUTDefinition(model="gemma", maker="google", driver="together")
    together_factory.make_sut(sut_definition, prefer_dedicated=False)
    assert together_factory.prefer_dedicated is False


def test_make_sut_not_found(together_factory):
    together_factory.client.chat.completions.create.side_effect = Exception("not serverless")
    together_factory.client.endpoints.list.return_value.data = []

    sut_definition = SUTDefinition(model="bogus", maker="fake", driver="together")
    with pytest.raises(ModelNotSupportedError):
        together_factory.make_sut(sut_definition)


@expensive_tests
def test_serverless_connection():
    factory = TogetherSUTFactory(load_secrets_from_config(path="."))
    sut_definition = SUTDefinition(maker="meta-llama", model="Llama-3.3-70B-Instruct-Turbo", driver="together")
    sut = factory.make_sut(sut_definition)
    assert sut.uid == "meta-llama/llama-3.3-70b-instruct-turbo:together"
    assert sut.model == "meta-llama/Llama-3.3-70B-Instruct-Turbo"
