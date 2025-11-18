import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from modelgauge.config import load_secrets_from_config
from modelgauge.dynamic_sut_factory import ModelNotSupportedError
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.anthropic_api import AnthropicSUT
from modelgauge.suts.anthropic_sut_factory import AnthropicSUTFactory


class FakeModel(dict):
    """A dict that pretends to be an object"""

    def __init__(self, *args, **kwargs):
        super(FakeModel, self).__init__(*args, **kwargs)
        self.__dict__ = self


class FakeModelsResponse(list):
    def __init__(self, json_response):
        super().__init__()
        for m in json_response["data"]:
            self.append(FakeModel(m))


@pytest.fixture
def factory():
    sut_factory = AnthropicSUTFactory({"anthropic": {"api_key": "value"}})
    mock_client = MagicMock()
    with open(Path(__file__).parent.parent / "data/anthropic-model-list.json", "r") as f:
        mock_client.models.list.return_value = FakeModelsResponse(json.load(f))
    sut_factory._client = mock_client

    return sut_factory


def test_make_sut(factory):
    sut_definition = SUTDefinition(model="claude-sonnet-4-5-20250929", driver="anthropic")
    sut = factory.make_sut(sut_definition)
    assert isinstance(sut, AnthropicSUT)
    assert sut.uid == "claude-sonnet-4-5-20250929:anthropic"
    assert sut.model == "claude-sonnet-4-5-20250929"
    assert sut.api_key == "value"


def test_make_sut_bad_model(factory):
    sut_definition = SUTDefinition(model="claude-bonnet-4-5-20250929", driver="anthropic")
    with pytest.raises(ModelNotSupportedError) as e:
        _ = factory.make_sut(sut_definition)
    assert "claude-sonnet-4-5-20250929" in str(e.value)


def test_autocorrect(factory):
    sut_definition = SUTDefinition(model="claude-sonnet-4-5", driver="anthropic")
    sut = factory.make_sut(sut_definition)
    assert isinstance(sut, AnthropicSUT)
    assert sut.uid == "claude-sonnet-4-5-20250929:anthropic"
    assert sut.model == "claude-sonnet-4-5-20250929"
    assert sut.api_key == "value"


def test_autocorrect_is_limited(factory):
    sut_definition = SUTDefinition(model="claude-sonnet", driver="anthropic")
    with pytest.raises(ModelNotSupportedError) as e:
        _ = factory.make_sut(sut_definition)
    assert "claude-sonnet-4-5-20250929" in str(e.value)


# @expensive_tests
def test_connection():
    factory = AnthropicSUTFactory(load_secrets_from_config(path="."))
    sut_definition = SUTDefinition(model="claude-sonnet-4-5-20250929", driver="anthropic")
    sut = factory.make_sut(sut_definition)
    assert sut.uid == "claude-sonnet-4-5-20250929:anthropic"
