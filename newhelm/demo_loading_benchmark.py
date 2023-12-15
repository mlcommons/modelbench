from typing import List
from newhelm.benchmark import BaseBenchmark
from newhelm.load_plugins import load_plugins
from newhelm.general import get_concrete_subclasses


if __name__ == "__main__":
    load_plugins()
    all_benchmarks: List[BaseBenchmark] = [
        cls() for cls in get_concrete_subclasses(BaseBenchmark)  # type: ignore[type-abstract]
    ]
    for benchmark in all_benchmarks:
        print(benchmark.__class__.__name__)
        for test in benchmark.get_tests():
            print("Uses test:", test.__class__.__name__)
