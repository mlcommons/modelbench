import faulthandler
import io
import json
import logging
import os
import pathlib
import pkgutil
import platform
import random
import signal
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from typing import List

import click
import termcolor
from click import echo
from rich.console import Console
from rich.table import Table

import modelgauge
from modelbench.benchmark_runner import BenchmarkRunner, TqdmRunTracker, JsonRunTracker
from modelbench.benchmarks import BenchmarkDefinition, GeneralPurposeAiChatBenchmarkV1
from modelbench.consistency_checker import ConsistencyChecker, summarize_consistency_check_results
from modelbench.hazards import STANDARDS
from modelbench.record import dump_json
from modelgauge.config import load_secrets_from_config, raise_if_missing_from_config, write_default_config
from modelgauge.load_plugins import load_plugins
from modelgauge.sut import SUT
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS
from modelgauge.tests.safe_v1 import PROMPT_SETS, Locale


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
    try:
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)
    except io.UnsupportedOperation:
        pass  # just an issue with some tests that capture sys.stderr

    log_dir = pathlib.Path("run/logs")
    log_dir.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        filename=log_dir / f'modelbench-{datetime.now().strftime("%y%m%d-%H%M%S")}.log', level=logging.INFO
    )
    write_default_config()
    load_plugins(disable_progress_bar=True)
    print()


@cli.command(help="run a benchmark")
@click.option(
    "--output-dir",
    "-o",
    default="./run/records",
    type=click.Path(file_okay=False, dir_okay=True, path_type=pathlib.Path),
)
@click.option("--max-instances", "-m", type=int, default=100)
@click.option("--debug", default=False, is_flag=True)
@click.option("--json-logs", default=False, is_flag=True, help="Print only machine-readable progress reports")
@click.option("sut_uids", "--sut", "-s", multiple=True, help="SUT uid(s) to run", required=True)
@click.option("--anonymize", type=int, help="Random number seed for consistent anonymization of SUTs")
@click.option("--parallel", default=False, help="Obsolete flag, soon to be removed")
@click.option(
    "--version",
    "-v",
    type=click.Choice(["1.0"]),
    default="1.0",
    help="Benchmark version to run (Default: 1.0)",
    multiple=False,
)
@click.option(
    "--locale",
    "-l",
    type=click.Choice(["en_us", "fr_fr"], case_sensitive=False),
    default="en_us",
    help=f"Locale for v1.0 benchmark (Default: en_us)",
    multiple=False,
)
@click.option(
    "--prompt-set",
    type=click.Choice(PROMPT_SETS.keys()),
    default="practice",
    help="Which prompt set to use",
    show_default=True,
)
@click.option(
    "--evaluator",
    type=click.Choice(["default", "ensemble"]),
    default="default",
    help="Which evaluator to use",
    show_default=True,
)
@local_plugin_dir_option
def benchmark(
    version: str,
    locale: str,
    output_dir: pathlib.Path,
    max_instances: int,
    debug: bool,
    json_logs: bool,
    sut_uids: List[str],
    anonymize=None,
    parallel=False,
    prompt_set="practice",
    evaluator="default",
) -> None:
    if parallel:
        click.echo("--parallel option unnecessary; benchmarks are now always run in parallel")
    start_time = datetime.now(timezone.utc)
    suts = find_suts_for_sut_argument(sut_uids)
    if locale == "all":
        locales = Locale
    else:
        locales = [Locale(locale)]

    benchmarks = [get_benchmark(version, l, prompt_set, evaluator) for l in locales]

    benchmark_scores = score_benchmarks(benchmarks, suts, max_instances, json_logs, debug)
    output_dir.mkdir(exist_ok=True, parents=True)
    for b in benchmarks:
        print_summary(b, benchmark_scores, anonymize)
        json_path = output_dir / f"benchmark_record-{b.uid}.json"
        scores = [score for score in benchmark_scores if score.benchmark_definition == b]
        dump_json(json_path, start_time, b, scores)
        print(f"Wrote record for {b.uid} to {json_path}.")
        # TODO: Consistency check


@cli.command(
    help="Check the consistency of a benchmark run using its journal file. You can pass the name of the file OR a directory containing multiple journal files (will be searched recursively)"
)
@click.argument("journal-path", type=click.Path(exists=True, dir_okay=True, path_type=pathlib.Path))
# @click.option("--record-path", "-r", type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path))
@click.option("--verbose", "-v", default=False, is_flag=True, help="Print details about the failed checks.")
def consistency_check(journal_path, verbose):
    journal_paths = []
    if journal_path.is_dir():
        # Search for all journal files in the directory.
        for p in journal_path.rglob("*"):
            if p.name.startswith("journal-run") and (p.suffix == ".jsonl" or p.suffix == ".zst"):
                journal_paths.append(p)
        if len(journal_paths) == 0:
            raise click.BadParameter(
                f"No journal files starting with 'journal-run' and ending with '.jsonl' or '.zst' found in the directory '{journal_path}'."
            )
    else:
        journal_paths = [journal_path]

    checkers = []
    checking_error_journals = []
    for p in journal_paths:
        echo(termcolor.colored(f"\nChecking consistency of journal {p} ..........", "green"))
        try:
            checker = ConsistencyChecker(p)
            checker.run(verbose)
            checkers.append(checker)
        except Exception as e:
            print("Error running consistency check", e)
            checking_error_journals.append(p)

    # Summarize results and unsuccessful checks.
    if len(checkers) > 1:
        echo(termcolor.colored("\nSummary of consistency checks for all journals:", "green"))
        summarize_consistency_check_results(checkers)
    if len(checking_error_journals) > 0:
        echo(termcolor.colored(f"\nCould not run checks on the following journals:", "red"))
        for j in checking_error_journals:
            print("\t", j)


def find_suts_for_sut_argument(sut_uids: List[str]):
    # TODO: Put object initialization code in once place shared with modelgauge.
    # Make sure we have all the secrets we need.
    secrets = load_secrets_from_config()
    missing_secrets = []
    unknown_uids = []
    suts = []
    for sut_uid in sut_uids:
        try:
            missing_secrets.extend(SUTS.get_missing_dependencies(sut_uid, secrets=secrets))
            suts.append(SUTS.make_instance(sut_uid, secrets=secrets))
        except KeyError:
            unknown_uids.append(sut_uid)
    if len(unknown_uids) > 0:
        valid_suts = sorted(SUTS.keys(), key=lambda x: x.lower())
        valid_suts_str = "\n\t".join(valid_suts)
        raise click.BadParameter(
            f"Unknown uids '{unknown_uids}'.\nValid options are: {valid_suts_str}",
            param_hint="sut",
        )
    raise_if_missing_from_config(missing_secrets)

    return suts


def ensure_ensemble_annotators_loaded():
    try:
        from modelgauge.private_ensemble_annotator_set import EnsembleAnnotatorSet, ensemble_secrets

        private_annotators = EnsembleAnnotatorSet(secrets=ensemble_secrets(load_secrets_from_config()))
        modelgauge.tests.safe_v1.register_private_annotator_tests(private_annotators, "ensemble")
        return True
    except Exception as e:
        warnings.warn(f"Can't load private ensemble annotators: {e}")
        return False


def get_benchmark(version: str, locale: Locale, prompt_set: str, evaluator) -> BenchmarkDefinition:
    if version == "0.5":
        raise ValueError("Version 0.5 is no longer supported.")
    elif version == "1.0":
        if evaluator == "ensemble":
            if not ensure_ensemble_annotators_loaded():
                print(f"Can't build benchmark for {str} {locale} {prompt_set} {evaluator}; couldn't load evaluator.")
                exit(1)
        return GeneralPurposeAiChatBenchmarkV1(locale, prompt_set, evaluator)
    else:
        raise ValueError(f"Unknown benchmark version: {version}")


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

    print(f"Starting run for {[b.uid for b in benchmarks]} over {[s.uid for s in suts]}")

    run = runner.run()

    print("Cache info:")
    print(run.cache_info())
    return run


@modelgauge_sut(capabilities=[])
class AnonSUT(SUT):
    pass


def print_summary(benchmark, benchmark_scores, anonymize):
    if anonymize:
        rng = random.Random(anonymize)
        rng.shuffle(benchmark_scores)

        counter = 0
        for bs in benchmark_scores:
            counter += 1
            uid = f"sut{counter:02d}"
            bs.sut = AnonSUT(uid)

    echo(termcolor.colored(f"\nBenchmarking complete for {benchmark.uid}.", "green"))
    console = Console()
    table = Table("SUT", "Grade", title="Overall Grades")
    for bs in benchmark_scores:
        table.add_row(bs.sut.uid, bs.grade_label())
        hazard_table = Table(
            "Hazard", "Grade", "# scored items", "# safe items", "# errors", title=f"Hazard Scores for SUT {bs.sut.uid}"
        )
        for hs in bs.hazard_scores:
            hazard_table.add_row(
                hs.hazard_definition.uid,
                hs.grade_label(),
                str(hs.num_scored_items),
                str(hs.num_safe_items),
                str(hs.exceptions),
            )
        console.print(hazard_table)
    console.print(table)


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
    reference_sut_uids = ["gemma-2-9b-it-hf", "llama-3.1-8b-instruct-turbo-together"]
    reference_suts = find_suts_for_sut_argument(reference_sut_uids)
    if not ensure_ensemble_annotators_loaded():
        print("Can't load private annotators needed for calibration")
        exit(1)

    benchmarks = []
    for l in [Locale.EN_US]:
        for prompt_set in PROMPT_SETS:
            benchmarks.append(GeneralPurposeAiChatBenchmarkV1(l, prompt_set, "ensemble"))
    run_result = run_benchmarks_for_suts(benchmarks, reference_suts, None)
    all_hazard_numeric_scores = defaultdict(list)
    for benchmark, scores_by_sut in run_result.benchmark_scores.items():
        for sut, benchmark_score in scores_by_sut.items():
            for hazard_score in benchmark_score.hazard_scores:
                all_hazard_numeric_scores[hazard_score.hazard_definition.uid].append(hazard_score.score.estimate)

    reference_standards = {h: min(s) for h, s in all_hazard_numeric_scores.items() if s}
    reference_standards = {k: reference_standards[k] for k in sorted(reference_standards.keys())}
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
            "reference_suts": [sut.uid for sut in reference_suts],
            "reference_standards": reference_standards,
        },
    }
    with open(standards_file, "w") as out:
        json.dump(result, out, indent=4)


if __name__ == "__main__":
    cli()
