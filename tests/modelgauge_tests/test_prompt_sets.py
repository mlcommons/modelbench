import pytest
from modelgauge.prompt_sets import prompt_set_file_base_name, PROMPT_SETS, TEST_PROMPT_SETS, validate_prompt_set


def test_file_base_name():
    assert prompt_set_file_base_name("bad") == ""
    assert prompt_set_file_base_name("fake-prompts") == "fake-prompts"
    assert prompt_set_file_base_name("practice") == PROMPT_SETS["practice"]


def test_validate_prompt_set():
    for s in PROMPT_SETS.keys():
        assert validate_prompt_set(s)
    for s in TEST_PROMPT_SETS.keys():
        assert validate_prompt_set(s)
    with pytest.raises(ValueError):
        validate_prompt_set("should raise")
