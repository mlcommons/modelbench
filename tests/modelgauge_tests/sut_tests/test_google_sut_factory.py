import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from modelgauge.config import load_secrets_from_config
from modelgauge.dynamic_sut_factory import ModelNotSupportedError
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.google_genai import GoogleGenAiSUT
from modelgauge.suts.google_sut_factory import GoogleSUTFactory
from modelgauge_tests.utilities import expensive_tests, FakeObject


class FakeModelsResponse(list):
    def __init__(self, json_response):
        super().__init__()
        for m in json_response["models"]:
            m["supported_actions"] = m["supportedGenerationMethods"]
            self.append(FakeObject(m))


@pytest.fixture
def factory():
    sut_factory = GoogleSUTFactory({"google_ai": {"api_key": "value"}})
    mock_gemini_client = MagicMock()
    with open(Path(__file__).parent.parent / "data/google-gemini-model-list.json", "r") as f:
        mock_gemini_client.models.list.return_value = FakeModelsResponse(json.load(f))
    sut_factory._gemini_client = mock_gemini_client

    return sut_factory


def test_sut_definition(factory):
    sut_definition = SUTDefinition(model="gemini-2.5-flash", driver="google")
    assert sut_definition.get("reasoning") is None


def test_make_sut(factory):
    sut_definition = SUTDefinition(model="gemini-2.5-flash", driver="google")
    with patch("modelgauge.suts.google_genai.genai.Client") as p:
        sut = factory.make_sut(sut_definition)
    assert isinstance(sut, GoogleGenAiSUT)
    assert sut.uid == "gemini-2.5-flash:google"
    assert sut.model_name == "gemini-2.5-flash"
    assert sut.client


def test_make_sut_bad_model(factory):
    sut_definition = SUTDefinition(model="gemini-2.6-flash", driver="google")
    with pytest.raises(ModelNotSupportedError) as e:
        _ = factory.make_sut(sut_definition)
    assert "gemini-2.5-flash" in str(e.value)


def test_list_suts(factory):
    suts = factory.list_suts()
    assert "google/gemini-2.5-flash:google" in [s.uid for s in suts]


@expensive_tests
def test_connection():
    factory = GoogleSUTFactory(load_secrets_from_config(path="."))
    sut_definition = SUTDefinition(model="gemini-2.5-flash", driver="google")
    sut = factory.make_sut(sut_definition)
    assert sut.uid == "gemini-2.5-flash:google"
