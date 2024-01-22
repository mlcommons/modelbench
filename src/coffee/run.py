import logging
import pathlib

import click

import coffee
from coffee.benchmark import (
    Benchmark,
    MakeshiftBiasBenchmark,
    MakeshiftToxicityBenchmark,
)
from coffee.helm_runner import HelmSut, BbqHelmTest, CliHelmRunner, InProcessHelmRunner
from coffee.static_site_generator import StaticSiteGenerator


def quantize_stars(raw_score):
    return round(2 * raw_score) / 2.0


@click.command()
@click.option(
    "--output-dir",
    "-o",
    default="./web",
    type=click.Path(file_okay=False, dir_okay=True, path_type=pathlib.Path),
)
@click.option("--max-instances", "-m", type=int, default=100)
@click.option("--debug", default=False, is_flag=True)
def cli(output_dir: pathlib.Path, max_instances: int, debug: bool) -> None:
    coffee.app_config.debug = debug

    if coffee.app_config.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    runner = InProcessHelmRunner()
    suts = [HelmSut.GPT2, HelmSut.PYTHIA_70M, HelmSut.FB_OPT_125M]

    benchmarks: list[Benchmark] = []
    for benchmark_class in [MakeshiftBiasBenchmark, MakeshiftToxicityBenchmark]:
        result = runner.run(benchmark_class.tests(), suts, max_instances=max_instances)
        if not result.success():
            print(
                f"HELM execution failed with return code {result.execution_result.returncode}:"
            )
            print("stdout:")
            print(result.helm_stdout())
            print("stderr:")
            print(result.helm_stderr())
        scores = result.load_scores()
        for sut in suts:
            benchmark = benchmark_class(sut, scores.for_sut(sut))
            benchmarks.append(benchmark)

    static_site_generator = StaticSiteGenerator()
    static_site_generator.generate(benchmarks, output_dir)


if __name__ == "__main__":
    cli()
