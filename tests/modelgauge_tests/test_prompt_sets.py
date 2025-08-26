import pytest
from modelgauge.prompt_sets import (
    GENERAL_PROMPT_SETS,
    prompt_set_file_base_name,
    prompt_set_from_url,
    validate_prompt_set,
)  # usort: skip


def test_file_base_name():
    assert prompt_set_file_base_name(GENERAL_PROMPT_SETS, "practice") == "airr_official_1.0_practice_prompt_set_release"
    assert (
        prompt_set_file_base_name(GENERAL_PROMPT_SETS, "practice", "en_us")
        == "airr_official_1.0_practice_prompt_set_release"
    )
    assert (
        prompt_set_file_base_name(GENERAL_PROMPT_SETS, "official", "fr_fr")
        == "airr_official_1.0_heldback_fr_fr_prompt_set_release"
    )

    with pytest.raises(ValueError):
        prompt_set_file_base_name(GENERAL_PROMPT_SETS, "bad")

    with pytest.raises(ValueError):
        prompt_set_file_base_name(GENERAL_PROMPT_SETS, "practice", "bogus")

    with pytest.raises(ValueError):
        prompt_set_file_base_name({"fake": "thing"}, "practice", "en_us")


def test_validate_prompt_set():
    for s in GENERAL_PROMPT_SETS.keys():
        assert validate_prompt_set(GENERAL_PROMPT_SETS, s, "en_us")
    with pytest.raises(ValueError):
        validate_prompt_set(GENERAL_PROMPT_SETS, "should raise")


def test_prompt_set_from_url():
    assert prompt_set_from_url("https://www.example.com/path/to/file.csv") == "file"
    assert prompt_set_from_url("https://www.example.com/thing.css") == "thing"
    assert prompt_set_from_url("degenerate string") == "degenerate string"
    assert prompt_set_from_url("https://www.example.com") == ""
    assert prompt_set_from_url("https://www.example.com/") == ""
