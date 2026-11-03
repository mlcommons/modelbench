import hashlib
import json
import os
import pathlib
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone


class NoStandardsFileError(Exception):
    def __init__(self, path: pathlib.Path):
        self.path = path
        super().__init__(
            f"Standards file {str(self.path)} does not exist. Please run `modelbench calibrate` on your desired benchmark."
        )


class OverwriteStandardsFileError(Exception):
    def __init__(self, path: pathlib.Path):
        self.path = path
        super().__init__(f"Error: attempting to overwrite existing standards file {str(self.path)}")


class JournalDigestMismatchError(Exception):
    """A calibration journal is missing, has no recorded digest, or its bytes differ from the recorded digest."""


def _sha256_of_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Standards:
    """Handles reading and writing of a standards file."""

    _FILE_FORMAT_VERSION = "2.0.0"

    def __init__(
        self,
        data: dict[str, float],
        reference_suts: list[str] | None = None,
        reference_benchmark: str | None = None,
        sut_scores: dict[str, dict[str, float]] | None = None,
        journals: list[dict[str, str] | str] | None = None,
    ):
        # Everything but data are optional so that we can read old standards files.
        # Everything is required if we are going to write a new standards file.
        self._data = data

        self._reference_suts = reference_suts
        self._reference_benchmark = reference_benchmark
        self._sut_scores = sut_scores
        self._journals = journals

    @classmethod
    def from_file(cls, path: pathlib.Path) -> "Standards":
        cls.assert_file_exists(path)
        with open(path) as f:
            data = json.load(f)
        standards = data["standards"]
        run_info = data["_metadata"]["run_info"]
        return cls(
            standards["reference_standards"],
            reference_suts=standards.get("reference_suts"),
            reference_benchmark=standards.get("reference_benchmark"),
            sut_scores=run_info.get("sut_scores"),
            journals=run_info.get("journals"),
        )

    @classmethod
    def from_runs(
        cls,
        reference_benchmark: "BenchmarkDefinition",
        sut_runs: dict["SUT", "BenchmarkRun"],
    ) -> "Standards":
        """Assumes that the runs have already been checked for consistency."""
        # First pull what we need out of the run structures.
        sut_scores = {}  # Maps SUT UID to a list of its float hazard scores
        journals = []
        for sut, run in sut_runs.items():
            scores = run.benchmark_scores[reference_benchmark][sut].hazard_scores
            sut_scores[sut.uid] = scores
            # Record the content digest with the name: a name alone cannot tell two runs apart.
            journals.append({"name": run.journal_path.name, "sha256": _sha256_of_file(run.journal_path)})

        sut_hazard_scores = defaultdict(dict)  # Maps sut UID to hazard UID to float score.
        scores_by_hazard = defaultdict(list)  # Maps hazard KEY to list of float scores
        for sut, hazard_scores in sut_scores.items():
            for hazard_score in hazard_scores:
                num_score = hazard_score.score.estimate
                assert (
                    hazard_score.hazard_definition.uid not in sut_hazard_scores[sut]
                ), f"Duplicate hazard {hazard_score.hazard_definition.uid} for SUT {sut}"
                sut_hazard_scores[sut][hazard_score.hazard_definition.uid] = num_score
                scores_by_hazard[hazard_score.hazard_definition.reference_key].append(num_score)

        # Check we have scores from all ref SUTs for each hazard.
        reference_suts = list(sut_scores.keys())
        assert reference_suts == reference_benchmark.reference_suts
        assert all(len(scores) == len(reference_suts) for scores in scores_by_hazard.values())

        reference_standards = {h: min(s) for h, s in scores_by_hazard.items()}
        return cls(
            reference_standards,
            reference_suts=reference_suts,
            reference_benchmark=reference_benchmark.uid,
            sut_scores=sut_hazard_scores,
            journals=journals,
        )

    def verify_journals(self, journal_dir: pathlib.Path) -> None:
        """Check that every calibration journal in journal_dir still has the bytes this file recorded.

        Raises JournalDigestMismatchError for a journal that is missing, was replaced by different bytes
        under the same name, or was recorded without a digest (standards files written before digests).
        """
        for entry in self._journals or []:
            if isinstance(entry, str):
                raise JournalDigestMismatchError(f"Journal {entry} has no recorded SHA-256, so it cannot be verified")
            path = journal_dir / entry["name"]
            if not path.is_file():
                raise JournalDigestMismatchError(f"Journal {entry['name']} is not in {journal_dir}")
            actual = _sha256_of_file(path)
            if actual != entry["sha256"]:
                raise JournalDigestMismatchError(
                    f"Journal {entry['name']} has SHA-256 {actual}, but the standards file recorded {entry['sha256']}"
                )

    @classmethod
    def get_standards_for_benchmark(cls, uid) -> "Standards":
        """This assumes uid is the reference benchmark."""
        path = cls._benchmark_standards_path(uid)
        return Standards.from_file(path)

    @classmethod
    def _benchmark_standards_path(cls, benchmark_uid: str) -> pathlib.Path:
        return pathlib.Path(__file__).parent / "standards" / f"{benchmark_uid}.json"

    @classmethod
    def assert_can_calibrate_benchmark(cls, benchmark: "BenchmarkDefinition"):
        """This assumes the benchmark is the reference benchmark."""
        cls.assert_file_does_not_exist(benchmark.uid)
        # Make sure all hazard keys are unique. Calibration logic cannot handle duplicate hazard keys.
        hazard_keys = [h.reference_key for h in benchmark.hazards()]
        if len(hazard_keys) != len(set(hazard_keys)):
            raise ValueError(
                f"Cannot calibrate reference {benchmark.uid} because it has duplicate hazard keys: {hazard_keys}. If multiple hazards in the benchmark share a reference score, then the benchmark must define a reference_benchmark that defines one hazard with that hazard key."
            )

    @classmethod
    def assert_file_does_not_exist(cls, benchmark_uid: str):
        path = cls._benchmark_standards_path(benchmark_uid)
        if path.exists():
            raise OverwriteStandardsFileError(path)

    @classmethod
    def assert_file_exists(cls, path: pathlib.Path):
        if not path.exists():
            raise NoStandardsFileError(path)

    @classmethod
    def assert_benchmark_standards_exist(cls, benchmark_uid: str):
        path = cls._benchmark_standards_path(benchmark_uid)
        cls.assert_file_exists(path)

    def reference_standard_for(self, hazard: "HazardDefinition") -> float:
        key = hazard.reference_key
        if key not in self._data:
            raise ValueError(
                f"Can't find standard for hazard UID {hazard.uid}. No hazard with key {key} in {self._data}"
            )
        return self._data[key]

    def dump_data(self):
        return json.dumps(self._data, indent=4)

    def write(self):
        self.assert_file_does_not_exist(self._reference_benchmark)
        reference_standards = dict(sorted(self._data.items()))  # Sort by hazard key.
        result = {
            "_metadata": {
                "NOTICE": f"This file is auto-generated by {sys.argv[0]}; avoid editing it manually.",
                "file_format_version": self._FILE_FORMAT_VERSION,
                "run_info": {
                    "user": os.environ.get("USER", os.environ.get("USERNAME")),
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z"),
                    "platform": platform.platform(),
                    "system": f"{platform.system()} {platform.release()} {platform.version()}",
                    "node": platform.node(),
                    "python": platform.python_version(),
                    "command": " ".join(sys.argv),
                    "sut_scores": self._sut_scores,
                    "journals": self._journals,
                },
            },
            "standards": {
                "reference_suts": self._reference_suts,
                "reference_benchmark": self._reference_benchmark,
                "reference_standards": reference_standards,
            },
        }

        path = self._benchmark_standards_path(self._reference_benchmark)
        with open(path, "w") as out:
            json.dump(result, out, indent=4)


class NullStandards:
    """A dummy class to fill in the standards object of uncalibrated benchmarks."""

    def reference_standard_for(self, hazard: "HazardDefinition"):
        return None
