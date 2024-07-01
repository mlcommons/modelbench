import json
import os
import platform
from datetime import datetime, timezone

from modelbench.benchmarks import BenchmarkScore, BenchmarkDefinition
from modelbench.hazards import HazardDefinition, HazardScore
from modelbench.modelgauge_runner import ModelGaugeSut
from modelbench.scoring import ValueEstimate


def benchmark_run_record(score):
    return {
        "score": score,
        "_metadata": {
            "format_version": 1,
            "run": {
                "user": os.environ.get("USER", os.environ.get("USERNAME")),
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z"),
                "platform": platform.platform(),
                "system": f"{platform.system()} {platform.release()} {platform.version()}",
                "node": platform.node(),
                "python": platform.python_version(),
            },
        },
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
            return o.key
        elif isinstance(o, ValueEstimate):
            return o.__dict__
        elif isinstance(o, datetime):
            return str(o)
        else:
            return super().default(o)
