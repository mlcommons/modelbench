import pytest
from modelgauge.prompt_sets import (
    PROMPT_SETS,
    prompt_set_file_base_name,
    validate_prompt_set,
)  # usort: skip


def test_file_base_name():
    assert prompt_set_file_base_name("practice") == "airr_official_1.0_practice_prompt_set_release"
    assert prompt_set_file_base_name("practice", "en_us") == "airr_official_1.0_practice_prompt_set_release"
    assert (
        prompt_set_file_base_name("practice", "en_us", PROMPT_SETS) == "airr_official_1.0_practice_prompt_set_release"
    )
    assert prompt_set_file_base_name("official", "fr_fr") == "airr_official_1.0_heldback_fr_fr_prompt_set_release"
    assert (
        prompt_set_file_base_name("official", "fr_fr", PROMPT_SETS)
        == "airr_official_1.0_heldback_fr_fr_prompt_set_release"
    )

    with pytest.raises(ValueError):
        prompt_set_file_base_name("bad")

    with pytest.raises(ValueError):
        prompt_set_file_base_name("practice", "bogus")

    with pytest.raises(ValueError):
        prompt_set_file_base_name("practice", "en_us", {"fake": "thing"})


def test_validate_prompt_set():
    for s in PROMPT_SETS.keys():
        assert validate_prompt_set(s, "en_us", PROMPT_SETS)
    with pytest.raises(ValueError):
        validate_prompt_set("should raise")
