import itertools
import logging
import pathlib
from collections import defaultdict

import click

import coffee
from coffee.benchmark import (
    GeneralChatBotBenchmarkDefinition,
    BenchmarkScore,
)
from coffee.helm import HelmSut, CliHelmRunner, HelmResult
from coffee.static_site_generator import StaticSiteGenerator


@click.command()
@click.option(
    "--output-dir",
    "-o",
    default="./web",
    type=click.Path(file_okay=False, dir_okay=True, path_type=pathlib.Path),
)
@click.option("--max-instances", "-m", type=int, default=100)
@click.option("--debug", default=False, is_flag=True)
@click.option("--web-only", default=False, is_flag=True)
def cli(output_dir: pathlib.Path, max_instances: int, debug: bool, web_only) -> None:
    coffee.app_config.debug = debug

    if coffee.app_config.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    runner = CliHelmRunner()
    suts = [HelmSut.GPT2, HelmSut.PYTHIA_70M, HelmSut.FB_OPT_125M]
    benchmark_scores = []
    for benchmark_definition in [GeneralChatBotBenchmarkDefinition()]:
        harm_scores_by_sut = defaultdict(list)
        for harm in benchmark_definition.harms():
            if web_only:
                # this is a little sketchy for now, a quick fix to make testing HTML changes easier
                tests = itertools.chain(
                    *[harm.tests() for harm in benchmark_definition.harms()]
                )
                result = HelmResult(list(tests), suts, pathlib.Path("./run"), None)
            else:
                result = runner.run(harm.tests(), suts, max_instances)
                if not result.success():
                    print(
                        f"HELM execution failed with return code {result.execution_result.returncode}:"
                    )
                    print("stdout:")
                    print(result.helm_stdout())
                    print("stderr:")
                    print(result.helm_stderr())

            helm_scores = result.load_scores()
            for sut in suts:
                harm_scores_by_sut[sut].append(harm.score(helm_scores.for_sut(sut)))
        for sut in suts:
            benchmark_scores.append(
                BenchmarkScore(benchmark_definition, sut, harm_scores_by_sut[sut])
            )

    static_site_generator = StaticSiteGenerator()
    static_site_generator.generate(benchmark_scores, output_dir)


if __name__ == "__main__":
    cli()
