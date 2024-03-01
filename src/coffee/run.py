import json
import logging
import os
import pathlib
import platform
import sys
from datetime import datetime, timezone
from typing import List, Mapping

import click
import newhelm
import termcolor
from click import echo
from newhelm.config import write_default_config, load_secrets_from_config
from newhelm.load_plugins import load_plugins
from newhelm.runners.simple_test_runner import run_prompt_response_test
from newhelm.sut_registry import SUTS

from coffee.benchmark import GeneralChatBotBenchmarkDefinition, BenchmarkScore, HazardDefinition, HazardScore, STANDARDS
from coffee.newhelm_runner import NewhelmSut
from coffee.static_site_generator import StaticSiteGenerator

_DEFAULT_SUTS = [
    NewhelmSut.GPT2,
    NewhelmSut.LLAMA_2_7B,
    NewhelmSut.LLAMA_2_13B,
    NewhelmSut.LLAMA_2_70B,
    NewhelmSut.MISTRAL_7B,
    NewhelmSut.PYTHIA_70M,
    NewhelmSut.YI_BASE_6B,
]


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
    write_default_config()
    load_secrets_from_config()
    load_plugins()


@cli.command(help="run the standard benchmark")
@click.option(
    "--output-dir", "-o", default="./web", type=click.Path(file_okay=False, dir_okay=True, path_type=pathlib.Path)
)
@click.option(
    "--max-instances", "-m", type=int, default=55
)  # this default is a hack to get a set that won't blow up in the toxicity annotator
@click.option("--debug", default=False, is_flag=True)
@click.option("--web-only", default=False, is_flag=True)
@click.option(
    "--sut",
    "-s",
    type=click.Choice([sut.key for sut in NewhelmSut]),
    multiple=True,
    default=[s.key for s in _DEFAULT_SUTS],
)
@click.option("--view-embed", default=False, is_flag=True, help="Render the HTML to be emdedded in another view")
def benchmark(
    output_dir: pathlib.Path, max_instances: int, debug: bool, web_only, sut: List[str], view_embed: bool
) -> None:
    suts = [s for s in NewhelmSut if s.key in sut]
    benchmark_scores = []
    benchmarks = [GeneralChatBotBenchmarkDefinition()]
    for sut in suts:
        echo(termcolor.colored(f'Examining system "{sut.display_name}"', "green"))
        sut_instance = SUTS.make_instance(sut.key)
        for benchmark_definition in benchmarks:
            echo(termcolor.colored(f'  Starting run for benchmark "{benchmark_definition.name()}"', "green"))
            hazard_scores = []
            for hazard in benchmark_definition.hazards():
                results = {}
                echo(termcolor.colored(f'    Examining hazard "{hazard.name()}"', "green"))

                if web_only:
                    # TODO load result from disk here
                    raise NotImplementedError
                else:
                    tests = hazard.tests()
                    counter = 0
                    for test in tests:
                        items = max_instances
                        if isinstance(test, newhelm.tests.bbq.BBQ):
                            # BBQ is currently multiple sub-tests, so roughly split the items among them
                            items = int(items / len(newhelm.tests.bbq._CATEGORIES))
                        results[test] = run_prompt_response_test(
                            f"test-{counter}", test, sut.key, sut_instance, "./run", items
                        )
                        counter += 1

                    score = hazard.score(results)
                    if debug:
                        echo(
                            termcolor.colored(
                                f"    For hazard {hazard.name()}, {sut.name} scores {score.value()}", "green"
                            )
                        )
                    hazard_scores.append(score)
            benchmark_scores.append(BenchmarkScore(benchmark_definition, sut, hazard_scores))

    echo()
    echo(termcolor.colored(f"Benchmarking complete, rendering reports...", "green"))
    static_site_generator = StaticSiteGenerator(view_embed=view_embed)
    static_site_generator.generate(benchmark_scores, output_dir)
    echo()
    echo(termcolor.colored(f"Reports complete, open {output_dir}/index.html", "green"))


@cli.command(help="Show and optionally update the benchmark three-star standard")
@click.option(
    "--update",
    default=False,
    is_flag=True,
    help="Run benchmarks for the reference sut and update the standard scores.",
)
@click.option(
    "--file",
    "-f",
    default=STANDARDS.path,
    type=click.Path(file_okay=True, dir_okay=False, path_type=pathlib.Path),
    help=f"Path to the the standards file you'd like to write; default is where the code looks: {STANDARDS.path}",
)
def calibrate(update: bool, file) -> None:
    echo("current standards")
    echo("-----------------")
    echo(json.dumps(STANDARDS.data, indent=4))

    if update:
        echo()
        update_standards_to(file)
        STANDARDS.reload()

        echo("new standards")
        echo("-------------")
        echo(json.dumps(STANDARDS.data, indent=4))


def update_standards_to(file):
    reference_sut = NewhelmSut.PYTHIA_70M
    hazards = GeneralChatBotBenchmarkDefinition().hazards()
    hazard_scores = run_tests(hazards, reference_sut, 100)
    result = {
        "_metadata": {
            "NOTICE": f"This file is auto-generated by {sys.argv[0]}; avoid editing it manually.",
            "run_info": {
                "user": os.environ.get("USER", os.environ.get("USERNAME")),
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z"),
                "platform": platform.platform(),
                "system": f"{platform.system()} {platform.release()} {platform.version()}",
                "node": platform.node(),
                "python": platform.python_version(),
            },
        },
        "standards": {
            "reference_sut": {"name": reference_sut.display_name, "id": reference_sut.key},
            "3_star": {hazard.__class__.name(): hazard_scores[hazard].value() for hazard in hazard_scores.keys()},
        },
    }
    with open(file, "w") as out:
        json.dump(result, out, indent=4)


def run_tests(hazards: List[HazardDefinition], sut: NewhelmSut, items: int) -> Mapping[HazardDefinition, HazardScore]:
    result = {}
    sut_instance = SUTS.make_instance(sut.key)
    for hazard in hazards:
        test_scores = {}
        for count, test in enumerate(hazard.tests()):
            test_scores[test] = run_prompt_response_test(f"test-{count}", test, sut.key, sut_instance, "./run", items)
        result[hazard] = hazard.score(test_scores)
    return result


if __name__ == "__main__":
    cli()
