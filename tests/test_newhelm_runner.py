import pathlib
import pickle

import pytest

from test_benchmark import SIMPLE_TOXICITY_DATA
from newhelm.tests.real_toxicity_prompts import RealToxicityPromptsResult


@pytest.mark.datafiles(SIMPLE_TOXICITY_DATA)
def test_read_scores(datafiles):
    with open(pathlib.Path(datafiles) / "test_records.pickle", "rb") as out:
        scores = pickle.load(out)

    test_records = list(scores.values())
    assert len(test_records) == 1
    result = test_records[0].result.to_instance()
    assert type(result) == RealToxicityPromptsResult
