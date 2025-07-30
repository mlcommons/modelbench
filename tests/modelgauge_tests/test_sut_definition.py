import pytest

from modelgauge.dynamic_sut_metadata import DynamicSUTMetadata
from modelgauge.sut_definition import SUTDefinition, SUTSpecification, SUTUIDGenerator


def test_convenience_methods():
    s = SUTSpecification()
    assert s.requires("model")
    assert not s.requires("reasoning")

    assert s.knows("moderated")
    assert not s.knows("bogus")


def test_frozen():
    definition = SUTDefinition()
    definition.add("model", "hello")
    definition.add("driver", "dolly")
    assert definition.uid == "hello:dolly"
    assert definition.dynamic_uid == "hello:dolly"
    with pytest.raises(AttributeError):
        definition.add("provider", "nebius")


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


def test_parse_rich_sut_uid():
    uid = "google/gemma-3-27b-it:nebius:hfrelay;mt=500;t=0.3"
    definition = SUTUIDGenerator.parse(uid)
    assert definition.validate()
    assert definition.get("model") == "gemma-3-27b-it"
    assert definition.get("maker") == "google"
    assert definition.get("driver") == "hfrelay"
    assert definition.get("provider") == "nebius"
    assert definition.get("max_tokens") == 500
    assert definition.get("temp") == 0.3


def test_identify_rich_sut_uids():
    assert SUTUIDGenerator.is_rich_sut_uid("google/gemma:vertexai;mt=1")
    assert SUTUIDGenerator.is_rich_sut_uid("google/gemma:vertexai")
    assert not SUTUIDGenerator.is_rich_sut_uid("gpt-4o")
