import json
import pathlib
from unittest.mock import patch

import pytest

from modelbench.benchmarks import BenchmarkDefinition
from modelbench.cli import calibrate, run_benchmarks_for_sut
from modelbench.hazards import HazardDefinition, HazardScore
from modelbench.scoring import ValueEstimate
from modelbench.standards import Standards
from modelgauge.prompt import TextPrompt
from modelgauge.single_turn_prompt_response import TestItem

from tests.modelbench_tests.test_benchmark_runner import AFakeTest
from tests.modelgauge_tests.fake_sut import FakeSUT


REFERENCE_SUTS = ["sut_1", "sut_2"]


class DummyHazard(HazardDefinition):
    def __init__(self, key):
        super().__init__()
        self._key = key
        self._tests = [AFakeTest("test1", [TestItem(prompt=TextPrompt(text="text"), source_id="id")])]

    def key(self):
        return self._key

    def test_uids(self) -> list[str]:
        return ["test1"]

    def score(self, sut_scores):
        return HazardScore(
            hazard_definition=self,
            score=ValueEstimate.make(0.5, 100),
            test_scores={},
            exceptions=0,
            num_scored_items=100,
            num_safe_items=5,
        )


class DummyBenchmark(BenchmarkDefinition):
    def __init__(self, standards, hazard, uid):
        self._standards = standards
        self._hazard = hazard
        self._uid = uid
        super().__init__()

    @property
    def standards(self):
        # Overloading this.
        return self._standards

    @property
    def reference_suts(self) -> list[str]:
        return REFERENCE_SUTS

    def reference_benchmark(self):
        return DummyBenchmark(self.standards, self._hazard, "reference_benchmark")

    def _make_hazards(self) -> list[HazardDefinition]:
        return [self._hazard]

    _uid_definition = {
        "uid": "self._uid",
    }


@pytest.fixture
def hazard():
    return DummyHazard(key="dummy_hazard")


@pytest.fixture
def standards(tmp_path, hazard):
    file = tmp_path / "benchmark_standards.json"
    standards = {"reference_suts": ["sut_1"], "reference_standards": {hazard.key(): 0.8}}

    with open(file, "w") as out:
        json.dump({"standards": standards}, out)

    return Standards(file)


def make_hazard_score(hazard: HazardDefinition, score: float):
    score = ValueEstimate(lower=score - 0.1, estimate=score, upper=score + 0.1, samples=100)
    return HazardScore(hazard_definition=hazard, score=score, test_scores={}, exceptions=0)


class TestStandards:
    def test_standards_read(self, standards, hazard):
        assert standards.path.exists()
        assert standards.reference_standard_for(hazard) == 0.8

    def test_standards_read_different_hazard_with_shared_key(self, standards, hazard):
        """Different hazards can share the same reference standard as long as they have the same key."""

        class AnotherHazard(HazardDefinition):
            def key(self):
                return hazard.key()

            def test_uids(self) -> list[str]:
                return []

            def score(self, sut_scores):
                pass

        another_hazard = AnotherHazard()
        assert standards.reference_standard_for(another_hazard) == standards.reference_standard_for(hazard)

    def test_standards_read_nonexistent_hazard(self, standards):
        nonexistent_hazard = DummyHazard(key="nonexistent_hazard")
        with pytest.raises(
            ValueError, match="Can't find standard for hazard UID dummy_hazard. No hazard with key nonexistent_hazard "
        ):
            standards.reference_standard_for(nonexistent_hazard)

    def test_standards_read_nonexistent_file(self):
        standards = Standards(pathlib.Path("nonexistent.json"))
        with pytest.raises(FileNotFoundError, match="Standards file nonexistent.json does not exist"):
            standards.reference_standard_for("dummy_hazard")

    def test_write_standards(self, tmp_path):
        """Multiple hazards and multiple SUTs."""
        path = tmp_path / "new_standards.json"
        standards = Standards(path)

        class DummyHazard2(DummyHazard):
            """Different hazard with different UID"""

            pass

        hazard_1 = DummyHazard(key="h1")
        hazard_2 = DummyHazard2(key="h2")

        standards.write_standards(
            {
                "sut_1": [make_hazard_score(hazard_1, 0.1), make_hazard_score(hazard_2, 0.9)],
                "sut_2": [make_hazard_score(hazard_1, 0.2), make_hazard_score(hazard_2, 0.8)],
            },
            DummyBenchmark(standards, hazard_1, "fake_benchmark"),
        )

        # Check can get new reference scores
        assert standards.reference_standard_for(hazard_1) == 0.1
        assert standards.reference_standard_for(hazard_2) == 0.8

        # Check actual file was written correctly
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["_metadata"]["run_info"]["sut_scores"] == {
            "sut_1": {"dummy_hazard": 0.1, "dummy_hazard_2": 0.9},
            "sut_2": {"dummy_hazard": 0.2, "dummy_hazard_2": 0.8},
        }
        assert data["standards"]["reference_suts"] == ["sut_1", "sut_2"]
        assert data["standards"]["reference_standards"] == {"h1": 0.1, "h2": 0.8}

    def test_overwrite_standards(self, standards, hazard):
        with pytest.raises(FileExistsError, match="Error: attempting to overwrite existing standards file"):
            standards.write_standards({}, DummyBenchmark(standards, hazard, "fake_benchmark"))

        # Check that the original standards file is unchanged
        assert standards.reference_standard_for(hazard) == 0.8


class TestCalibration:
    @patch("modelbench.cli.make_sut")
    def test_calibrate(self, mock_sut, tmp_path, hazard):
        """Make sure the correct benchmark gets run and the correct number gets written when we run calibrate."""
        # TODO: make it so this doesn't write to the actual run directory.
        sut_1 = FakeSUT(REFERENCE_SUTS[0])
        sut_2 = FakeSUT(REFERENCE_SUTS[1])
        mock_sut.side_effect = [sut_1, sut_2]

        path = tmp_path / "new_standards.json"
        standards = Standards(path)
        benchmark = DummyBenchmark(standards, hazard, "fake_benchmark")
        with patch("modelbench.cli.run_benchmarks_for_sut", wraps=run_benchmarks_for_sut) as mock_run:
            calibrate(benchmark)
            # Should be called once per SUT
            assert mock_run.call_count == 2
            # Make sure the right benchmark was ran both times.
            for call in mock_run.call_args_list:
                args, kwargs = call
                assert len(args[0]) == 1
                assert args[0][0].uid == "reference_benchmark"

        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["standards"]["reference_suts"] == REFERENCE_SUTS
        assert data["standards"]["reference_benchmark"] == "reference_benchmark"
        # The reference standard is the smaller of the two scores
        assert data["standards"]["reference_standards"] == {"dummy_hazard": 0.5}

    # @patch("modelbench.cli.make_sut")
    # def test_calibrate_fails_with_bad_run(self, mock_sut, tmp_path, hazard):
    # Put back when we add consistency checks for calibration runs
    #     class FailingSUT(FakeSUT):
    #         def evaluate(self, request):
    #             raise RuntimeError("Simulated evaluation failure")
    #
    #     sut_1 = FailingSUT(TestCalibration.reference_suts[0])
    #     sut_2 = FakeSUT(TestCalibration.reference_suts[1])
    #
    #     mock_sut.side_effect = [sut_1, sut_2]
    #
    #     path = tmp_path / "new_standards.json"
    #     standards = Standards(path)
    #     benchmark = self.DummyBenchmark(standards, hazard)
    #     with pytest.raises(ValueError, match=f"Consistency check failed for reference SUT {TestCalibration.reference_suts[0]}. Can't calibrate standards."):
    #         calibrate(benchmark)
    #     # Make sure nothing was written
    #     assert not path.exists()
