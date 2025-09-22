import json

from modelgauge.sut_definition import SUTUIDGenerator, SUTDefinition


def test_uid():
    d = SUTDefinition()
    d.add("model", "chatgpt-4o")
    d.add("maker", "openai")
    d.add("driver", "openai")
    d.add("provider", "openai")
    d.add("date", None)  # workaround for the test in test_to_dynamic_sut_metadata setting the date (?!)
    d.add("moderated", True)
    d.add("reasoning", False)
    g = SUTUIDGenerator(d)
    assert g.uid == "openai/chatgpt-4o:openai:openai;mod=y;reas=n"

    d.add("date", "20250723")
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
