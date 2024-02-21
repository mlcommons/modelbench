import json
import pathlib
from unittest.mock import patch

from coffee.benchmark import HarmScore, BiasHarmDefinition
from coffee.run import update_standards_to


@patch("coffee.run.run_tests")
def test_update_standards(fake_run, tmp_path):
    bias_harm = BiasHarmDefinition()
    fake_run.return_value = {bias_harm: HarmScore(bias_harm, 0.123456)}
    new_path = pathlib.Path(tmp_path) / "standards.json"
    update_standards_to(new_path)
    assert new_path.exists()
    with open(new_path) as f:
        j = json.load(f)
        assert j["standards"]["3_star"][bias_harm.name()] == 0.123456
