from modelgauge.dynamic_sut_metadata import DynamicSUTMetadata
from modelgauge.uid_generator import SUTUIDGenerator


def test_uid():
    g = SUTUIDGenerator()
    g.add("model", "chatgpt-4o")
    g.add("maker", "openai")
    g.add("driver", "openai")
    g.add("provider", "openai")
    g.add("temperature", 0.8)
    g.add("top_p", 6)
    g.add("top_k", 5)
    g.add("moderated", True)
    g.add("reasoning", False)
    assert g.uid == "openai/chatgpt-4o:openai:openai_mod:y_reas:n_t:0.8_p:6_k:5"

    g.add("driver_code_version", "abcde")
    g.add("date", "20250723")
    assert g.uid == "openai/chatgpt-4o:openai:openai:20250723_dv:abcde_mod:y_reas:n_t:0.8_p:6_k:5"

    g.add("display_name", "my favorite SUT")
    assert g.uid == "openai/chatgpt-4o:openai:openai:20250723_dv:abcde_mod:y_reas:n_t:0.8_p:6_k:5_dn:my-favorite-sut"


def test_from_json():
    data = {"model": "the_model", "driver": "the_driver"}
    g = SUTUIDGenerator.from_json(data)
    assert g.data["model"] == "the_model"
    assert g.data["driver"] == "the_driver"

    data_s = '{"model": "my_model", "driver": "my_driver"}'
    gg = SUTUIDGenerator.from_json(data_s)
    assert gg.data["model"] == "my_model"
    assert gg.data["driver"] == "my_driver"


def test_to_dynamic_sut_metadata():
    data = {
        "model": "the_model",
        "driver": "the_driver",
        "maker": "the_maker",
        "provider": "the_provider",
        "date": "20250724",
    }
    g = SUTUIDGenerator(data)
    assert g.to_dynamic_sut_metadata() == DynamicSUTMetadata(**data)
