from typing import List
from newhelm.benchmark import BaseBenchmark
from newhelm.load_plugins import load_plugins
from newhelm.general import get_concrete_subclasses, to_json
from newhelm.plugins.runners.simple_benchmark_runner import SimpleBenchmarkRunner
from newhelm.sut import SUT


if __name__ == "__main__":
    load_plugins()
    all_benchmarks: List[BaseBenchmark] = [
        cls() for cls in get_concrete_subclasses(BaseBenchmark)  # type: ignore[type-abstract]
    ]
    all_suts: List[SUT] = [
        cls() for cls in get_concrete_subclasses(SUT)  # type: ignore[type-abstract]
    ]
    runner = SimpleBenchmarkRunner()
    for benchmark in all_benchmarks:
        print("\n\nStarting:", benchmark.__class__.__name__)
        benchmark_journals = runner.run(benchmark, all_suts)
        for journal in benchmark_journals:
            # make it print pretty
            print(to_json(journal, indent=4))
