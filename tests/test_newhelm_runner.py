import pathlib
import pickle

import pytest

from test_benchmark import SIMPLE_CHAT_DATA
from newhelm.tests.real_toxicity_prompts import RealToxicityPromptsResult


@pytest.mark.datafiles(SIMPLE_CHAT_DATA)
def test_read_scores(datafiles):
    with open(pathlib.Path(datafiles) / "test_records.pickle", "rb") as out:
        scores = pickle.load(out)

    result = scores["real_toxicity_prompts"].result.to_instance()
    assert type(result) == RealToxicityPromptsResult
