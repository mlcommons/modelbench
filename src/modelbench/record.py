import json
import os
import pathlib
import platform
import subprocess
from datetime import datetime, timezone
from typing import Sequence

import pydantic
from modelgauge.base_test import BaseTest

from modelbench.benchmarks import BenchmarkScore, BenchmarkDefinition
from modelbench.hazards import HazardDefinition, HazardScore
from modelbench.static_site_generator import StaticContent
from modelbench.suts import ModelGaugeSut, SutDescription


def run_command(*args):
    result = subprocess.run(args, capture_output=True)
    return result.stdout.decode("utf-8").strip()


def benchmark_code_info():
    try:
        git_dir = run_command("git", "rev-parse", "git-dir")
        if not git_dir:
            return {"error": "couldn't find git dir"}
    except FileNotFoundError:
        return {"error": "git command not found"}

    return {
        "git_version": run_command("git", "--version"),
        "origin": run_command("git", "config", "--get", "remote.origin.url"),
        "code_version": run_command("git", "describe", "--tags", "--abbrev=8", "--always", "--long", "--match", "v*"),
        "changed_files": [l.strip() for l in run_command("git", "status", "-s", "--untracked-files=no").splitlines()],
    }


def benchmark_library_info():
    try:
        text = run_command("python", "-m", "pip", "list")
        result = {}
        for line in text.splitlines()[2:]:
            package, version = line.split(maxsplit=1)
            result[package] = version
        return result
    except FileNotFoundError:
        return {"error": "pip not found"}


def benchmark_metadata():
    return {
        "format_version": 1,
        "run": {
            "user": os.environ.get("USER", os.environ.get("USERNAME")),
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z"),
            "platform": platform.platform(),
            "system": f"{platform.system()} {platform.release()} {platform.version()}",
            "node": platform.node(),
            "python": platform.python_version(),
        },
        "code": {
            "source": benchmark_code_info(),
            "libraries": benchmark_library_info(),
        },
    }


def benchmark_run_record(score):
    return {
        "score": score,
        "_metadata": benchmark_metadata(),
    }


def dump_json(
    json_path: pathlib.Path,
    start_time: datetime.time,
    benchmark: BenchmarkDefinition,
    benchmark_scores: Sequence[BenchmarkScore],
):
    with open(json_path, "w") as f:
        output = {
            "benchmark": (benchmark),
            "run_uid": f"run-{benchmark.uid}-{start_time.strftime('%Y%m%d-%H%M%S')}",
            "scores": (benchmark_scores),
            "content": StaticContent(),
        }
        json.dump(output, f, cls=BenchmarkScoreEncoder, indent=4)


class BenchmarkScoreEncoder(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, BenchmarkScore) or isinstance(o, HazardScore):
            result = {}
            result.update(o.__dict__)
            result["numeric_grade"] = o.numeric_grade()
            result["text_grade"] = o.text_grade()
            if "benchmark_definition" in result:
                del result["benchmark_definition"]  # duplicated up the tree
            return result
        elif isinstance(o, BenchmarkDefinition):
            return {"uid": o.uid, "hazards": o.hazards()}
        elif isinstance(o, HazardDefinition):
            result = {"uid": o.uid, "reference_standard": o.reference_standard()}
            if o._tests:
                result["tests"] = o._tests
            return result
        elif isinstance(o, BaseTest):
            return o.uid
        elif isinstance(o, SutDescription):
            result = {"uid": o.key}
            if isinstance(o, ModelGaugeSut) and o.instance_initialization():
                result["initialization"] = o.instance_initialization()
            return result
        elif isinstance(o, pydantic.BaseModel):
            return o.model_dump()
        elif isinstance(o, datetime):
            return str(o)
        else:
            return super().default(o)
