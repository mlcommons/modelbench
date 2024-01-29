from newhelm.benchmark_registry import BENCHMARKS
from newhelm.command_line import newhelm_cli

from newhelm.load_plugins import load_plugins
from newhelm.sut_registry import SUTS
from newhelm.test_registry import TESTS


@newhelm_cli.command()
def list():
    print("SUTS:")
    for sut, entry in SUTS.items():
        print("\t", sut, entry)
    print("Tests:")
    for test, entry in TESTS.items():
        print("\t", test, entry)
    print("Benchmarks:")
    for benchmark, entry in BENCHMARKS.items():
        print("\t", benchmark, entry)


if __name__ == "__main__":
    load_plugins()
    newhelm_cli()
