from newhelm.benchmark_registry import BENCHMARKS
from newhelm.command_line import newhelm_cli

from newhelm.load_plugins import load_plugins, list_plugins
from newhelm.sut_registry import SUTS
from newhelm.test_registry import TESTS


@newhelm_cli.command()
def list():
    plugins = list_plugins()
    print(f"Plugin Modules: {len(plugins)}")
    for module_name in plugins:
        print("\t", module_name)
    suts = SUTS.items()
    print(f"SUTS: {len(suts)}")
    for sut, entry in suts:
        print("\t", sut, entry)
    tests = TESTS.items()
    print(f"Tests: {len(tests)}")
    for test, entry in tests:
        print("\t", test, entry)
    benchmarks = BENCHMARKS.items()
    print(f"Benchmarks: {len(benchmarks)}")
    for benchmark, entry in benchmarks:
        print("\t", benchmark, entry)


if __name__ == "__main__":
    load_plugins()
    newhelm_cli()
