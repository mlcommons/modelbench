import json
import pathlib
from unittest.mock import patch

from coffee.benchmark import HazardScore, BiasHazardDefinition
from coffee.run import update_standards_to


@patch("coffee.run.run_tests")
def test_update_standards(fake_run, tmp_path):
    bias_hazard = BiasHazardDefinition()
    fake_run.return_value = {bias_hazard: HazardScore(bias_hazard, 0.123456)}
    new_path = pathlib.Path(tmp_path) / "standards.json"
    update_standards_to(new_path)
    assert new_path.exists()
    with open(new_path) as f:
        j = json.load(f)
        assert j["standards"]["3_star"][bias_hazard.name()] == 0.123456
        assert j["standards"]["reference_sut"]["id"] == "pythia-70m"
