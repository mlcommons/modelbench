from modelgauge.prompt_sets import PROMPT_SETS, PromptSet


def test_find():
    prompt_sets = [
        PromptSet(prompt_set_type="official", locale="hi_hi", version="1.0", filename="some_file1.csv"),
        PromptSet(prompt_set_type="heldback", locale="en_us", version="0.5", filename="some_file2.csv"),
        PromptSet(prompt_set_type="official", locale="en_us", version="1.0", filename="some_file3.csv"),
        PromptSet(prompt_set_type="official", locale="fr_fr", version="1.1", filename="some_file4.csv"),
    ]
    assert PromptSet.find("official", "hi_hi", "1.0", prompt_sets) == prompt_sets[0]
    assert PromptSet.find("heldback", "en_us", "0.5", prompt_sets) == prompt_sets[1]
    assert PromptSet.find("official", "en_us", "1.0", prompt_sets) == prompt_sets[2]
    assert PromptSet.find("official", "fr_fr", "1.1", prompt_sets) == prompt_sets[3]
    assert PromptSet.find("fake", "fr_fr", "2.0", prompt_sets) is None

    # the blessed PROMPT_SETS must always include en_us official and heldback 1.0
    assert PromptSet.find("practice", "en_us", "1.0", PROMPT_SETS) is not None
    assert PromptSet.find("official", "en_us", "1.0", PROMPT_SETS) is not None
    assert PromptSet.find("practice", "fr_fr", "1.0", PROMPT_SETS) is not None


def test_url():
    ps = PromptSet(prompt_set_type="official", locale="hi_hi", version="1.0", filename="some_file1.csv")
    assert ps.url() == "https://ailuminate.mlcommons.org/files/download/some_file1.csv"
