import logging
import pathlib

import click
import newhelm
import termcolor
from newhelm.general import get_or_create_json_file
from newhelm.runners.simple_benchmark_runner import run_prompt_response_test
from newhelm.sut_registry import SUTS

from coffee.benchmark import GeneralChatBotBenchmarkDefinition, BenchmarkScore
from coffee.newhelm_runner import NewhelmSut
from coffee.static_site_generator import StaticSiteGenerator


def _make_output_dir():
    o = pathlib.Path.cwd()
    if o.name in ["src", "test"]:
        logging.warning(f"Output directory of {o} looks suspicious")
    if not o.name == "run":
        o = o / "run"
    o.mkdir(exist_ok=True)
    return o


@click.group()
def cli() -> None:
    import newhelm.load_plugins

    newhelm.load_plugins.load_plugins()
    from newhelm.secrets_registry import SECRETS

    SECRETS.set_values(get_or_create_json_file("secrets/default.json"))


@cli.command()
@click.option(
    "--output-dir", "-o", default="./web", type=click.Path(file_okay=False, dir_okay=True, path_type=pathlib.Path)
)
@click.option(
    "--max-instances", "-m", type=int, default=55
)  # this default is a hack to get a set that won't blow up in the toxicity annotator
@click.option("--debug", default=False, is_flag=True)
@click.option("--web-only", default=False, is_flag=True)
def benchmark(output_dir: pathlib.Path, max_instances: int, debug: bool, web_only) -> None:
    suts = [NewhelmSut.GPT2, NewhelmSut.LLAMA_2_7B]
    benchmark_scores = []
    benchmarks = [GeneralChatBotBenchmarkDefinition()]
    for sut in suts:
        print(termcolor.colored(f'Examining system "{sut.display_name}"', "yellow"))
        for benchmark_definition in benchmarks:
            print(termcolor.colored(f'  Starting run for benchmark "{benchmark_definition.name()}"', "green"))
            print(f"Benchmark definition: {benchmark_definition}")
            harm_scores = []
            for harm in benchmark_definition.harms():
                results = {}
                print(termcolor.colored(f'    Examining harm "{harm.name()}"', "yellow"))

                if web_only:
                    # TODO load result from disk here
                    raise NotImplementedError
                else:
                    tests = harm.tests()
                    counter = 0
                    for test in tests:
                        items = max_instances
                        if isinstance(test, newhelm.tests.bbq.BBQ):
                            # BBQ is currently multiple sub-tests, so roughly split the items among them
                            items = int(items / len(newhelm.tests.bbq._CATEGORIES))
                        results[test] = run_prompt_response_test(
                            f"test-{counter}", test, sut.key, SUTS.make_instance(sut.key), "./run", items
                        )
                        counter += 1

                    score = harm.score(results)
                    if debug:
                        print(
                            termcolor.colored(
                                f"    For harm {harm.name()}, {sut.name} scores {score.value()}", "yellow"
                            )
                        )
                    harm_scores.append(score)
            benchmark_scores.append(BenchmarkScore(benchmark_definition, sut, harm_scores))

    print()
    print(termcolor.colored(f"Benchmarking complete, rendering reports...", "green"))
    static_site_generator = StaticSiteGenerator()
    static_site_generator.generate(benchmark_scores, output_dir)
    print()
    print(termcolor.colored(f"Reports complete, open {output_dir}/index.html", "green"))


if __name__ == "__main__":
    cli()
