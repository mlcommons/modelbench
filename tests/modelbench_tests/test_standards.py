import json
import pathlib
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from modelbench.benchmarks import BenchmarkDefinition
from modelbench.cli import calibrate, run_benchmarks_for_sut
from modelbench.hazards import HazardDefinition, HazardScore
from modelbench.scoring import ValueEstimate
from modelbench.standards import NoStandardsFileError, OverwriteStandardsFileError, Standards
from modelgauge.annotators import fake_annotator
from modelgauge.prompt import TextPrompt
from modelgauge.single_turn_prompt_response import TestItem

from tests.modelbench_tests.test_benchmark_runner import AFakeTest
from tests.modelgauge_tests.fake_sut import FakeSUT


REFERENCE_SUTS = ["sut_1", "sut_2"]


class DummyTest(AFakeTest):
    def __init__(self, uid: str, items, annotators=["fake_annotator"]):
        super().__init__(uid, items, annotators)

    def measure_quality(self, item):
        return {"is_safe": 0.0}

    def aggregate_measurements(self, items):
        return {"is_safe": 0.0}


class DummyHazard(HazardDefinition):
    def __init__(self, key):
        super().__init__()
        self._key = key
        self._tests = [DummyTest("test1", [TestItem(prompt=TextPrompt(text="text"), source_id="id")])]

    @property
    def reference_key(self):
        return self._key

    def test_uids(self) -> list[str]:
        return ["test1"]

    def score(self, sut_scores):
        return HazardScore(
            hazard_definition=self,
            score=ValueEstimate.make(0.0, 100),
            test_scores={},
            exceptions=0,
            num_scored_items=100,
            num_safe_items=0,
        )


class DummyHazard2(DummyHazard):
    """Different hazard with different UID"""

    pass


class DummyBenchmark(BenchmarkDefinition):
    def __init__(self, hazards, uid):
        self._hazards = hazards
        self._uid = uid
        super().__init__()

    @property
    def reference_suts(self) -> list[str]:
        return REFERENCE_SUTS

    def reference_benchmark(self):
        return DummyBenchmark(self._hazards, "reference_benchmark")

    def _make_hazards(self) -> list[HazardDefinition]:
        return self._hazards

    _uid_definition = {
        "uid": "self._uid",
    }


@pytest.fixture
def hazard():
    return DummyHazard(key="dummy_hazard")


@pytest.fixture
def standards(hazard):
    return Standards({hazard.reference_key: 0.8})


@pytest.fixture
def standards_path_patch(monkeypatch, tmp_path):
    path = tmp_path / "standards.json"
    monkeypatch.setattr(
        Standards,
        "_benchmark_standards_path",
        classmethod(lambda cls, uid: path),
    )
    return path


def make_standards_from_runs(hazard_1, hazard_2):
    # 2 SUTs, 2 hazards
    benchmark = DummyBenchmark([hazard_1, hazard_2], "fake_benchmark")
    sut_1 = FakeSUT(REFERENCE_SUTS[0])
    sut_2 = FakeSUT(REFERENCE_SUTS[1])

    class SimpleRun(BaseModel):
        benchmark_scores: dict
        journal_path: pathlib.Path = pathlib.Path("journal.json")

    class HazardScores(BaseModel):
        hazard_scores: list

    sut_runs = {
        sut_1: SimpleRun(
            benchmark_scores={
                benchmark: {
                    sut_1: HazardScores(
                        hazard_scores=[make_hazard_score(hazard_1, 0.1), make_hazard_score(hazard_2, 0.9)]
                    )
                }
            }
        ),
        sut_2: SimpleRun(
            benchmark_scores={
                benchmark: {
                    sut_2: HazardScores(
                        hazard_scores=[make_hazard_score(hazard_1, 0.2), make_hazard_score(hazard_2, 0.8)]
                    )
                }
            }
        ),
    }
    return Standards.from_runs(benchmark, sut_runs)


def make_hazard_score(hazard: HazardDefinition, score: float):
    score = ValueEstimate(lower=score - 0.1, estimate=score, upper=score + 0.1, samples=100)
    return HazardScore(hazard_definition=hazard, score=score, test_scores={}, exceptions=0)


class TestStandards:
    def test_from_file(self, tmp_path, hazard):
        file = tmp_path / "benchmark_standards.json"
        standards = {
            "reference_standards": {hazard.reference_key: 0.8},
            "reference_suts": REFERENCE_SUTS,
            "reference_benchmark": "reference_benchmark",
        }
        sut_scores = {REFERENCE_SUTS[0]: {hazard.uid: 0.1}, REFERENCE_SUTS[1]: {hazard.uid: 0.2}}
        run_info = {"sut_scores": sut_scores, "journals": ["journal.json"]}
        with open(file, "w") as out:
            json.dump({"standards": standards, "_metadata": {"run_info": run_info}}, out)
        standards = Standards.from_file(file)
        assert standards.reference_standard_for(hazard) == 0.8
        assert standards._reference_suts == REFERENCE_SUTS
        assert standards._reference_benchmark == "reference_benchmark"
        assert standards._sut_scores == sut_scores
        assert standards._journals == ["journal.json"]

    def test_from_file_old_format(self, tmp_path, hazard):
        """Test we can still read files with the old format."""
        file = tmp_path / "benchmark_standards.json"
        standards = {"reference_standards": {hazard.reference_key: 0.8}}
        with open(file, "w") as out:
            json.dump({"standards": standards, "_metadata": {"run_info": {}}}, out)

        standards = Standards.from_file(file)
        assert standards.reference_standard_for(hazard) == 0.8

    def test_from_runs(self):
        hazard_1 = DummyHazard(key="h1")
        hazard_2 = DummyHazard2(key="h2")

        standards = make_standards_from_runs(hazard_1, hazard_2)

        assert standards.reference_standard_for(hazard_1) == 0.1
        assert standards.reference_standard_for(hazard_2) == 0.8

    def test_get_standards_for_benchmark(self, tmp_path, hazard, standards_path_patch):
        benchmark = DummyBenchmark([hazard], "fake_benchmark")
        standards = {"reference_standards": {hazard.reference_key: 0.8}}
        with open(standards_path_patch, "w") as out:
            json.dump({"standards": standards, "_metadata": {"run_info": {}}}, out)

        standards = Standards.get_standards_for_benchmark(benchmark.uid)
        assert standards.reference_standard_for(hazard) == 0.8

    def test_get_standards_for_benchmark_no_file(self):
        with pytest.raises(NoStandardsFileError):
            Standards.get_standards_for_benchmark("nonexistent_benchmark")

    def test_assert_benchmark_standards_exist(self, tmp_path, monkeypatch):
        with pytest.raises(NoStandardsFileError):
            Standards.assert_benchmark_standards_exist("nonexistent_benchmark")

        existing_benchmark = "existing_benchmark"
        monkeypatch.setattr(
            Standards,
            "_benchmark_standards_path",
            classmethod(lambda cls, uid: tmp_path / f"{uid}.json"),
        )
        file = Standards._benchmark_standards_path(existing_benchmark)
        with open(file, "w") as out:
            json.dump({"standards": {"reference_standards": {}}, "_metadata": {"run_info": {}}}, out)

        # Nothing should be raised
        Standards.assert_benchmark_standards_exist(existing_benchmark)

    def test_standards_read_different_hazard_with_shared_key(self, standards, hazard):
        """Different hazards can share the same reference standard as long as they have the same key."""

        class AnotherHazard(HazardDefinition):
            @property
            def reference_key(self):
                return hazard.reference_key

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

    def test_write(self, tmp_path, standards_path_patch):
        hazard_1 = DummyHazard(key="h1")
        hazard_2 = DummyHazard2(key="h2")

        standards = make_standards_from_runs(hazard_1, hazard_2)
        standards.write()

        assert standards_path_patch.exists()
        with open(standards_path_patch) as f:
            data = json.load(f)
        assert data["_metadata"]["run_info"]["sut_scores"] == {
            "sut_1": {"dummy_hazard": 0.1, "dummy_hazard_2": 0.9},
            "sut_2": {"dummy_hazard": 0.2, "dummy_hazard_2": 0.8},
        }
        assert data["_metadata"]["run_info"]["journals"] == ["journal.json", "journal.json"]
        assert data["standards"]["reference_suts"] == ["sut_1", "sut_2"]
        assert data["standards"]["reference_standards"] == {"h1": 0.1, "h2": 0.8}

    def test_overwrite_standards(self, tmp_path, standards_path_patch):
        standards_path_patch.write_text("original standards")

        standards = Standards({}, reference_benchmark="fake_benchmark")
        with pytest.raises(OverwriteStandardsFileError):
            standards.write()

        # Check that the original standards file is unchanged
        assert standards_path_patch.read_text() == "original standards"


class TestCalibration:
    @patch("modelbench.cli.make_sut")
    def test_calibrate(self, mock_sut, tmp_path, hazard, standards_path_patch):
        """Make sure the correct benchmark gets run and the correct number gets written when we run calibrate."""
        sut_1 = FakeSUT(REFERENCE_SUTS[0])
        sut_2 = FakeSUT(REFERENCE_SUTS[1])
        mock_sut.side_effect = [sut_1, sut_2]

        benchmark = DummyBenchmark([hazard], "fake_benchmark")
        with patch("modelbench.cli.run_benchmarks_for_sut", wraps=run_benchmarks_for_sut) as mock_run:
            calibrate(benchmark, run_path=str(tmp_path))
            # Should be called once per SUT
            assert mock_run.call_count == 2
            # Make sure the right benchmark was run both times.
            for call in mock_run.call_args_list:
                args, kwargs = call
                assert len(args[0]) == 1
                assert args[0][0].uid == "reference_benchmark"

        assert standards_path_patch.exists()
        with open(standards_path_patch) as f:
            data = json.load(f)
        assert len(data["_metadata"]["run_info"]["journals"]) == 2
        assert all(["journal" in j for j in data["_metadata"]["run_info"]["journals"]])
        assert data["standards"]["reference_suts"] == REFERENCE_SUTS
        assert data["standards"]["reference_benchmark"] == "reference_benchmark"
        # The reference standard is the smaller of the two scores
        assert data["standards"]["reference_standards"] == {"dummy_hazard": 0.0}

    def test_calibrate_fails_duplicate_hazard_keys(self, tmp_path, hazard):
        """Make sure that calibration fails up front if the benchmark has multiple hazards with the same reference key."""
        benchmark = DummyBenchmark([hazard, hazard], "fake_benchmark")
        with pytest.raises(ValueError, match="Cannot calibrate"):
            calibrate(benchmark, run_path=str(tmp_path))

    @patch("modelbench.cli.make_sut")
    def test_calibrate_fails_with_bad_run(self, mock_sut, tmp_path, hazard, standards_path_patch):
        class FailingSUT(FakeSUT):
            def __init__(self, uid):
                super().__init__(uid)
                self._calls = 0

            def evaluate(self, request):
                # First call should succeed so that it passed the initial sut check.
                if self._calls == 0:
                    self._calls += 1
                    return super().evaluate(request)
                raise RuntimeError("Simulated evaluation failure")

        sut_1 = FailingSUT(REFERENCE_SUTS[0])
        sut_2 = FakeSUT(REFERENCE_SUTS[1])
        mock_sut.side_effect = [sut_1, sut_2]

        benchmark = DummyBenchmark([hazard], "fake_benchmark")
        with pytest.raises(
            RuntimeError,
            match=f"Consistency check failed for reference SUT {REFERENCE_SUTS[0]}. Standards not updated.",
        ):
            calibrate(benchmark, run_path=str(tmp_path))
        # Make sure nothing was written
        assert not standards_path_patch.exists()
