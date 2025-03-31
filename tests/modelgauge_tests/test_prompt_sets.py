import pytest
from modelgauge.prompt_sets import (
    PROMPT_SETS,
    demo_prompt_set_from_private_prompt_set,
    demo_prompt_set_url,
    prompt_set_file_base_name,
    prompt_set_from_url,
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


def test_demo_prompt_set_from_private_prompt_set():
    assert demo_prompt_set_from_private_prompt_set(PROMPT_SETS["practice"]["en_us"]) == PROMPT_SETS["demo"]["en_us"]
    assert demo_prompt_set_from_private_prompt_set(PROMPT_SETS["practice"]["fr_fr"]) == PROMPT_SETS["demo"]["fr_fr"]
    assert demo_prompt_set_from_private_prompt_set(PROMPT_SETS["official"]["en_us"]) == PROMPT_SETS["demo"]["en_us"]
    assert demo_prompt_set_from_private_prompt_set(PROMPT_SETS["official"]["fr_fr"]) == PROMPT_SETS["demo"]["fr_fr"]
    assert demo_prompt_set_from_private_prompt_set(PROMPT_SETS["demo"]["en_us"]) == PROMPT_SETS["demo"]["en_us"]
    assert demo_prompt_set_from_private_prompt_set(PROMPT_SETS["demo"]["fr_fr"]) == PROMPT_SETS["demo"]["fr_fr"]
    assert demo_prompt_set_from_private_prompt_set("bogus") == "bogus"


def test_prompt_set_from_url():
    assert prompt_set_from_url("https://www.example.com/path/to/file.csv") == "file"
    assert prompt_set_from_url("https://www.example.com/thing.css") == "thing"
    assert prompt_set_from_url("degenerate string") == "degenerate string"
    assert prompt_set_from_url("https://www.example.com") == ""
    assert prompt_set_from_url("https://www.example.com/") == ""


def test_demo_prompt_set_url():
    base = "https://www.example.com/path/to/"
    for l in ("en_us", "fr_fr"):
        for t in ("practice", "official"):
            base_url = f"{base}{PROMPT_SETS[t][l]}.csv"
            assert demo_prompt_set_url(base_url) == f"{base}{PROMPT_SETS['demo'][l]}.csv"
