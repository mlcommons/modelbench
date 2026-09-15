import pytest
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.openai_client import OpenAIResponsesSUT
from modelgauge.suts.openai_sut_factory import OpenAICompatibleSUTFactory
from modelgauge.suts.openrouter_sut_factory import OPENROUTER_BASE_URL


@pytest.fixture
def factory():
    return OpenAICompatibleSUTFactory(
        raw_secrets={
            "openrouter": {"api_key": "some_key"},
        }
    )


def test_factory_makes_correct_openrouter_sut(factory):
    sut_definition = SUTDefinition(model="gpt-oss-20b", maker="openai", driver="openai", provider="openrouter")
    sut = factory.make_sut(sut_definition)
    assert isinstance(sut, OpenAIResponsesSUT)
    assert sut.uid == "openai/gpt-oss-20b:openrouter:openai"
    assert sut.model == "gpt-oss-20b"
    assert str(sut.client.base_url).startswith(OPENROUTER_BASE_URL)
