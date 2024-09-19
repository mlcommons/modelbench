import itertools
import json
import os
import pathlib
import pkgutil
import platform
import random
import sys
from datetime import datetime, timezone
from typing import List, Optional

import click
import termcolor
from click import echo
from modelgauge.config import load_secrets_from_config, write_default_config
from modelgauge.load_plugins import load_plugins
from modelgauge.sut_registry import SUTS

from modelbench.benchmark_runner import BenchmarkRunner, TqdmRunTracker, JsonRunTracker
from modelbench.benchmarks import (
    BenchmarkDefinition,
)
from modelbench.hazards import STANDARDS
from modelbench.record import dump_json
from modelbench.static_site_generator import StaticContent, StaticSiteGenerator
from modelbench.suts import ModelGaugeSut, SutDescription, SUTS_FOR_V_0_5

_DEFAULT_SUTS = SUTS_FOR_V_0_5


def load_local_plugins(_, __, path: pathlib.Path):
    path_str = str(path)
    sys.path.append(path_str)
    plugins = pkgutil.walk_packages([path_str])
    for plugin in plugins:
        __import__(plugin.name)


local_plugin_dir_option = click.option(
    "--plugin-dir",
    type=click.Path(exists=True, dir_okay=True, path_type=pathlib.Path, file_okay=False),
    help="Directory containing plugins to load",
    callback=load_local_plugins,
    expose_value=False,
)


@click.group()
@local_plugin_dir_option
def cli() -> None:
    write_default_config()
    load_plugins(disable_progress_bar=True)
    print()
    print(StaticContent()["general"]["provisional_disclaimer"])
    print()


@cli.command(help="run a benchmark")
@click.option(
    "--output-dir", "-o", default="./web", type=click.Path(file_okay=False, dir_okay=True, path_type=pathlib.Path)
)
@click.option("--max-instances", "-m", type=int, default=100)
@click.option("--debug", default=False, is_flag=True)
@click.option("--json-logs", default=False, is_flag=True, help="Print only machine-readable progress reports")
@click.option("sut_uids", "--sut", "-s", multiple=True, help="SUT uid(s) to run")
@click.option("--view-embed", default=False, is_flag=True, help="Render the HTML to be embedded in another view")
@click.option(
    "--custom-branding",
    type=click.Path(file_okay=False, dir_okay=True, exists=True, path_type=pathlib.Path),
    help="Path to directory containing custom branding.",
)
@click.option("--anonymize", type=int, help="Random number seed for consistent anonymization of SUTs")
@click.option("--parallel", default=False, help="Obsolete flag, soon to be removed")
@click.option(
    "benchmark_name",
    "--benchmark",
    type=click.Choice([c.__name__ for c in BenchmarkDefinition.__subclasses__()]),
    default="GeneralPurposeAiChatBenchmark",
    help="Benchmark to run (Default: GeneralPurposeAiChatBenchmark)",
    multiple=False,
)
@local_plugin_dir_option
def benchmark(
    benchmark_name: str,
    output_dir: pathlib.Path,
    max_instances: int,
    debug: bool,
    json_logs: bool,
    sut_uids: List[str],
    view_embed: bool,
    custom_branding: Optional[pathlib.Path] = None,
    anonymize=None,
    parallel=False,
) -> None:
    if parallel:
        click.echo("--parallel option unnecessary; benchmarks are now always run in parallel")
    start_time = datetime.now(timezone.utc)
    suts = find_suts_for_sut_argument(sut_uids)
    benchmark = BenchmarkDefinition.find_by_name(benchmark_name)
    benchmark_scores = score_benchmarks([benchmark], suts, max_instances, json_logs, debug)
    generate_content(benchmark_scores, output_dir, anonymize, view_embed, custom_branding)
    json_path = output_dir / f"benchmark_record-{benchmark.uid}.json"
    dump_json(json_path, start_time, benchmark, benchmark_scores)


def find_suts_for_sut_argument(sut_args: List[str]):
    if sut_args:
        suts = []
        default_suts_by_key = {s.key: s for s in SUTS_FOR_V_0_5}
        registered_sut_keys = set(i[0] for i in SUTS.items())
        for sut_arg in sut_args:
            if sut_arg in default_suts_by_key:
                suts.append(default_suts_by_key[sut_arg])
            elif sut_arg in registered_sut_keys:
                suts.append(ModelGaugeSut.for_key(sut_arg))
            else:
                all_sut_keys = registered_sut_keys.union(set(default_suts_by_key.keys()))
                raise click.BadParameter(
                    f"Unknown key '{sut_arg}'. Valid options are {sorted(all_sut_keys, key=lambda x: x.lower())}",
                    param_hint="sut",
                )

    else:
        suts = SUTS_FOR_V_0_5
    return suts


def score_benchmarks(benchmarks, suts, max_instances, json_logs=False, debug=False):
    run = run_benchmarks_for_suts(benchmarks, suts, max_instances, debug=debug, json_logs=json_logs)

    benchmark_scores = []
    for bd, score_dict in run.benchmark_scores.items():
        for k, score in score_dict.items():
            benchmark_scores.append(score)
    return benchmark_scores


def run_benchmarks_for_suts(benchmarks, suts, max_instances, debug=False, json_logs=False, thread_count=32):
    runner = BenchmarkRunner(pathlib.Path("./run"))
    runner.secrets = load_secrets_from_config()
    runner.benchmarks = benchmarks
    runner.suts = suts
    runner.max_items = max_instances
    runner.debug = debug
    runner.thread_count = thread_count
    runner.run_tracker = JsonRunTracker() if json_logs else TqdmRunTracker(0.5)
    run = runner.run()
    return run


class FakeSut(SutDescription):
    @property
    def name(self):
        return self.key.upper()


def generate_content(benchmark_scores, output_dir, anonymize, view_embed, custom_branding=None):
    static_site_generator = StaticSiteGenerator(view_embed=view_embed, custom_branding=custom_branding)
    if anonymize:

        rng = random.Random(anonymize)
        rng.shuffle(benchmark_scores)

        counter = 0
        for bs in benchmark_scores:
            counter += 1
            key = f"sut{counter:02d}"
            name = f"System Under Test {counter}"

            bs.sut = FakeSut(key, name)
            static_site_generator._content[key] = {"name": name, "tagline": "A well-known model."}
    echo(termcolor.colored(f"\nBenchmarking complete, rendering reports...", "green"))
    static_site_generator.generate(benchmark_scores, output_dir)
    echo(termcolor.colored(f"\nReports complete, open {output_dir}/index.html", "green"))


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
    if file is not STANDARDS.path:
        STANDARDS.path = file
        STANDARDS.reload()
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


def update_standards_to(standards_file):
    reference_suts = [
        ModelGaugeSut.for_key("mistral-7b"),
    ]
    run_result = run_benchmarks_for_suts([c() for c in BenchmarkDefinition.__subclasses__()], reference_suts, None)
    hazards = set(itertools.chain.from_iterable([b.hazards() for b in run_result.benchmarks]))
    all_hazard_numeric_scores = {h.uid: [] for h in hazards}
    for benchmark, scores_by_sut in run_result.benchmark_scores.items():
        for sut, benchmark_score in scores_by_sut.items():
            for hazard_score in benchmark_score.hazard_scores:
                all_hazard_numeric_scores[hazard_score.hazard_definition.uid].append(hazard_score.score.estimate)

    reference_standards = {h: min(s) for h, s in all_hazard_numeric_scores.items() if s}
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
            "reference_suts": [sut.key for sut in reference_suts],
            "reference_standards": reference_standards,
        },
    }
    with open(standards_file, "w") as out:
        json.dump(result, out, indent=4)


if __name__ == "__main__":
    cli()
