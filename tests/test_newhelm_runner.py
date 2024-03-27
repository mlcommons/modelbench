import pathlib
import pickle

import pytest
from newhelm.tests.safe import SafeTestResult

from test_benchmark import SIMPLE_CAE_DATA


@pytest.mark.datafiles(SIMPLE_CAE_DATA)
def test_read_scores(datafiles):
    with open(pathlib.Path(datafiles) / "test_records.pickle", "rb") as out:
        scores = pickle.load(out)

    test_records = list(scores.values())
    assert len(test_records) == 2
    result = test_records[0].result.to_instance()
    assert type(result) == SafeTestResult
