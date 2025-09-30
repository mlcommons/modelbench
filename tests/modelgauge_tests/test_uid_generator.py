import json

from modelgauge.sut_definition import SUTUIDGenerator, SUTDefinition


def test_uid():
    raw = {
        "model": "chatgpt-4o",
        "maker": "openai",
        "driver": "openai",
        "provider": "openai",
        "moderated": True,
        "reasoning": False,
    }
    g = SUTUIDGenerator(SUTDefinition(raw))
    assert g.uid == "openai/chatgpt-4o:openai:openai;mod=y;reas=n"

    raw["date"] = "20250723"
    g = SUTUIDGenerator(SUTDefinition(raw))
    assert g.uid == "openai/chatgpt-4o:openai:openai:20250723;mod=y;reas=n"


def test_is_json_string():
    data = {"this": 1, "that": "other"}
    s = json.dumps(data)
    assert SUTUIDGenerator.is_json_string(s)
    s = "hello"
    assert not SUTUIDGenerator.is_json_string(s)


def test_is_file(tmpdir_factory):
    f = tmpdir_factory.mktemp("data").join("data.json")
    data = {"this": 1, "that": "other"}
    with open(f, "w") as of:
        of.write(json.dumps(data))
    assert SUTUIDGenerator.is_file(str(f))
    assert not SUTUIDGenerator.is_file(json.dumps(data))
    assert not SUTUIDGenerator.is_file("not a file name")
