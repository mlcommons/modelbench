import pytest

from modelgauge.prompt_sets import PROMPT_SETS, PromptSet


def test_find():
    prompt_sets = [
        PromptSet(prompt_set_type="official", locale="hi_hi", filename="some_file1.csv"),
        PromptSet(prompt_set_type="heldback", locale="en_us", filename="some_file2.csv"),
        PromptSet(prompt_set_type="official", locale="en_us", filename="some_file3.csv"),
        PromptSet(prompt_set_type="official", locale="fr_fr", filename="some_file4.csv"),
    ]
    assert PromptSet.find("official", "hi_hi", prompt_sets) == prompt_sets[0]
    assert PromptSet.find("heldback", "en_us", prompt_sets) == prompt_sets[1]
    assert PromptSet.find("official", "en_us", prompt_sets) == prompt_sets[2]
    assert PromptSet.find("official", "fr_fr", prompt_sets) == prompt_sets[3]
    assert PromptSet.find("fake", "fr_fr", prompt_sets) is None


def test_url():
    ps = PromptSet(prompt_set_type="official", locale="hi_hi", filename="some_file1.csv")
    assert ps.url() == "https://ailuminate.mlcommons.org/files/download/some_file1.csv"
