import pytest
from modelgauge.prompt_sets import (
    PROMPT_SETS,
    TEST_PROMPT_SETS,
    prompt_set_file_base_name,
    validate_prompt_set,
)  # usort: skip


def test_file_base_name():
    assert prompt_set_file_base_name("bad") is None
    assert prompt_set_file_base_name("fake-prompts", TEST_PROMPT_SETS) == "fake-prompts"
    assert prompt_set_file_base_name("practice") == PROMPT_SETS["practice"]
    assert prompt_set_file_base_name("practice", PROMPT_SETS) == PROMPT_SETS["practice"]


def test_validate_prompt_set():
    for s in PROMPT_SETS.keys():
        assert validate_prompt_set(s, PROMPT_SETS)
    for s in TEST_PROMPT_SETS.keys():
        assert validate_prompt_set(s, TEST_PROMPT_SETS)
    with pytest.raises(ValueError):
        validate_prompt_set("should raise")
