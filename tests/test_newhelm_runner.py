import pathlib
import pickle

import pytest

from test_benchmark import SIMPLE_TOXICITY_DATA


@pytest.mark.datafiles(SIMPLE_TOXICITY_DATA)
def test_read_scores(datafiles):
    with open(pathlib.Path(datafiles) / "test_records.pickle", "rb") as out:
        scores = pickle.load(out)

    test_records = list(scores.values())
    assert len(test_records) == 1
    results = test_records[0].results
    assert len(results) == 3
    assert results[2].name == "empirical_probability_toxicity"
