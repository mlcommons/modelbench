import pytest

from modelgauge.dynamic_sut_metadata import DynamicSUTMetadata
from modelgauge.sut_definition import SUTDefinition, SUTSpecification, SUTUIDGenerator


@pytest.fixture
def empty_definition():
    d = SUTDefinition()
    return d


@pytest.fixture
def definition():
    data = {"model": "the_model", "driver": "the_driver"}
    d = SUTDefinition(data)
    return d


def test_convenience_methods():
    s = SUTSpecification()
    assert s.requires("model")
    assert not s.requires("reasoning")

    assert s.knows("moderated")
    assert not s.knows("bogus")


def test_frozen():
    d = SUTDefinition()
    d.add("model", "hello")
    d.add("driver", "dolly")
    assert d.uid == "hello:dolly"
    assert d.dynamic_uid == "hello:dolly"
    with pytest.raises(AttributeError):
        d.add("provider", "nebius")


def test_from_json():
    data_s = '{"model": "my_model", "driver": "my_driver"}'
    dd = SUTDefinition.from_json_string(data_s)
    assert dd.get("model") == "my_model"
    assert dd.get("driver") == "my_driver"


def test_to_dynamic_sut_metadata():
    data = {
        "model": "the_model",
        "driver": "the_driver",
        "maker": "the_maker",
        "provider": "the_provider",
        "date": "20250724",
    }
    d = SUTDefinition(data)
    assert d.to_dynamic_sut_metadata() == DynamicSUTMetadata(**data)


def test_parse_rich_sut_uid(empty_definition):
    uid = "google/gemma-3-27b-it:nebius:hfrelay;mt=500;t=0.3"
    empty_definition = SUTUIDGenerator.parse(uid)
    assert empty_definition.validate()
    assert empty_definition.get("model") == "gemma-3-27b-it"
    assert empty_definition.get("maker") == "google"
    assert empty_definition.get("driver") == "hfrelay"
    assert empty_definition.get("provider") == "nebius"
    assert empty_definition.get("max_tokens") == 500
    assert empty_definition.get("temp") == 0.3


def test_identify_rich_sut_uids():
    assert SUTUIDGenerator.is_rich_sut_uid("google/gemma:vertexai;mt=1")
    assert SUTUIDGenerator.is_rich_sut_uid("google/gemma:vertexai")
    assert not SUTUIDGenerator.is_rich_sut_uid("gpt-4o")
