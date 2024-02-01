import click

from newhelm.benchmark_registry import BENCHMARKS
from newhelm.command_line import newhelm_cli

from newhelm.load_plugins import load_plugins, list_plugins
from newhelm.sut_registry import SUTS
from newhelm.test_registry import TESTS


@newhelm_cli.command()
def list() -> None:
    plugins = list_plugins()
    click.echo(click.style(f"Plugin Modules: {len(plugins)}", bold=True))
    for module_name in plugins:
        click.echo("\t", nl=False)
        click.echo(module_name)
    suts = SUTS.items()
    click.echo(click.style(f"SUTS: {len(suts)}", bold=True))
    for sut, sut_entry in suts:
        click.echo("\t", nl=False)
        click.echo(f"{sut} {sut_entry}")
    tests = TESTS.items()
    click.echo(click.style(f"Tests: {len(tests)}", bold=True))
    for test, test_entry in tests:
        click.echo("\t", nl=False)
        click.echo(f"{test} {test_entry}")
    benchmarks = BENCHMARKS.items()
    click.echo(click.style(f"Benchmarks: {len(benchmarks)}", bold=True))
    for benchmark, benchmark_entry in benchmarks:
        click.echo("\t", nl=False)
        click.echo(f"{benchmark} {benchmark_entry}")


if __name__ == "__main__":
    load_plugins()
    newhelm_cli()
