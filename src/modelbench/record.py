import json
import os
import platform
import subprocess
from datetime import datetime, timezone

import pydantic

from modelbench.benchmarks import BenchmarkScore, BenchmarkDefinition
from modelbench.hazards import HazardDefinition, HazardScore
from modelbench.modelgauge_runner import ModelGaugeSut


def run_command(*args):
    result = subprocess.run(args, capture_output=True)
    return result.stdout.decode("utf-8").strip()


def benchmark_code_info():
    try:
        return {
            "git_version": run_command("git", "--version"),
            "origin": run_command("git", "config", "--get", "remote.origin.url"),
            "code_version": run_command(
                "git", "describe", "--tags", "--abbrev=8", "--always", "--long", "--match", "v*"
            ),
            "changed_files": [
                l.strip() for l in run_command("git", "status", "-s", "--untracked-files=no").splitlines()
            ],
        }
    except FileNotFoundError:
        return {"error": "git command not found"}


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
        "code": benchmark_code_info(),
    }


def benchmark_run_record(score):
    return {
        "score": score,
        "_metadata": benchmark_metadata(),
    }


class BenchmarkScoreEncoder(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, BenchmarkScore) or isinstance(o, HazardScore):
            result = {}
            result.update(o.__dict__)
            result["numeric_grade"] = o.numeric_grade()
            result["text_grade"] = o.text_grade()
            return result
        elif isinstance(o, BenchmarkDefinition):
            return {"uid": o.uid, "hazards": o.hazards()}
        elif isinstance(o, HazardDefinition):
            return o.uid
        elif isinstance(o, ModelGaugeSut):
            result = {"uid": o.key}
            if o.instance_initialization():
                result["initialization"] = o.instance_initialization()
            return result
        elif isinstance(o, pydantic.BaseModel):
            return o.__dict__
        elif isinstance(o, datetime):
            return str(o)
        else:
            return super().default(o)
