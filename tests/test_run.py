import json
import pathlib
import unittest.mock
from unittest.mock import patch

from modelbench.scoring import ValueEstimate
from modelbench.hazards import HazardScore, SafeCbrHazard
from modelbench.run import update_standards_to


@patch("modelbench.run.run_tests")
def test_update_standards(fake_run, tmp_path, fake_secrets):
    with unittest.mock.patch("modelbench.run.load_secrets_from_config", return_value=fake_secrets):
        bias_hazard = SafeCbrHazard()
        fake_run.return_value = {
            bias_hazard: HazardScore(
                hazard_definition=bias_hazard, score=ValueEstimate.make(0.123456, 100), test_scores={}
            )
        }
        new_path = pathlib.Path(tmp_path) / "standards.json"
        update_standards_to(new_path)
        assert new_path.exists()
        with open(new_path) as f:
            j = json.load(f)
            assert j["standards"]["reference_standards"][bias_hazard.key()] == 0.123456
            assert j["standards"]["reference_suts"][0]["id"] == "llama-2-13b-chat"
