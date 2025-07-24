from modelgauge.dynamic_sut_metadata import DynamicSUTMetadata
from modelgauge.sut_specification import SUTDefinition, SUTSpecification


def test_convenience_methods():
    s = SUTSpecification()
    assert s.requires("model")
    assert not s.requires("reasoning")

    assert s.knows("moderated")
    assert not s.knows("bogus")


def test_from_json():
    data = {"model": "the_model", "driver": "the_driver"}
    d = SUTDefinition(data)
    assert d.get("model") == "the_model"
    assert d.get("driver") == "the_driver"

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
