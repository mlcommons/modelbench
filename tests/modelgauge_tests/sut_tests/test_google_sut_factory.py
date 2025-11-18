import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from modelgauge.config import load_secrets_from_config
from modelgauge.dynamic_sut_factory import ModelNotSupportedError
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.google_genai import GoogleGenAiSUT
from modelgauge.suts.google_sut_factory import GoogleSUTFactory
from modelgauge_tests.utilities import expensive_tests


class FakeModel(dict):
    """A dict that pretends to be an object"""

    def __init__(self, *args, **kwargs):
        super(FakeModel, self).__init__(*args, **kwargs)
        self.__dict__ = self


class FakeModelsResponse(list):
    def __init__(self, json_response):
        super().__init__()
        for m in json_response["models"]:
            self.append(FakeModel(m))


@pytest.fixture
def factory():
    sut_factory = GoogleSUTFactory({"google_ai": {"api_key": "value"}})
    mock_gemini_client = MagicMock()
    with open(Path(__file__).parent.parent / "data/google-gemini-model-list.json", "r") as f:
        mock_gemini_client.models.list.return_value = FakeModelsResponse(json.load(f))
    sut_factory._gemini_client = mock_gemini_client

    return sut_factory


def test_make_sut(factory):
    sut_definition = SUTDefinition(model="gemini-2.5-flash", driver="google")
    sut = factory.make_sut(sut_definition)
    assert isinstance(sut, GoogleGenAiSUT)
    assert sut.uid == "gemini-2.5-flash:google"
    assert sut.model_name == "gemini-2.5-flash"
    assert sut.api_key == "value"


def test_make_sut_bad_model(factory):
    sut_definition = SUTDefinition(model="gemini-2.6-flash", driver="google")
    with pytest.raises(ModelNotSupportedError) as e:
        _ = factory.make_sut(sut_definition)
    assert "gemini-2.5-flash" in str(e.value)


@expensive_tests
def test_connection():
    factory = GoogleSUTFactory(load_secrets_from_config(path="."))
    sut_definition = SUTDefinition(model="gemini-2.5-flash", driver="google")
    sut = factory.make_sut(sut_definition)
    assert sut.uid == "gemini-2.5-flash:google"
