import json
import pathlib
import unittest.mock
from unittest.mock import patch

from coffee.benchmark import Hazard, HazardScore, ValueEstimate
from coffee.run import update_standards_to


@patch("coffee.run.run_tests")
def test_update_standards(fake_run, tmp_path, fake_secrets):
    with unittest.mock.patch("coffee.run.load_secrets_from_config", return_value=fake_secrets):

        fake_run.return_value = [HazardScore(hazard=Hazard.BIAS, score=ValueEstimate.make([0.123456]), test_scores={})]
        new_path = pathlib.Path(tmp_path) / "standards.json"
        update_standards_to(new_path)
        assert new_path.exists()
        with open(new_path) as f:
            j = json.load(f)
            assert j["standards"]["3_star"][str(Hazard.BIAS)] == 0.123456
            assert j["standards"]["reference_sut"]["id"] == "pythia-70m"
