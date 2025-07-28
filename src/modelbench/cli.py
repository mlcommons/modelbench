import datetime
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
from collections import defaultdict
from datetime import datetime, timezone

import click

import termcolor
from click import echo
from modelgauge.config import load_secrets_from_config, write_default_config
from modelgauge.load_plugins import load_plugins
from modelgauge.locales import DEFAULT_LOCALE, LOCALES, PUBLISHED_LOCALES, validate_locale
from modelgauge.monitoring import PROMETHEUS
from modelgauge.preflight import check_secrets, make_sut
from modelgauge.prompt_sets import PROMPT_SETS, validate_prompt_set
from modelgauge.sut import SUT
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS

from rich.console import Console
from rich.table import Table

from modelbench.benchmark_runner import BenchmarkRunner, JsonRunTracker, TqdmRunTracker
from modelbench.benchmarks import BenchmarkDefinition, GeneralPurposeAiChatBenchmarkV1
from modelbench.consistency_checker import ConsistencyChecker, summarize_consistency_check_results
from modelbench.hazards import STANDARDS
from modelbench.record import dump_json


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
    PROMETHEUS.push_metrics()
    try:
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)
    except io.UnsupportedOperation:
        pass  # just an issue with some tests that capture sys.stderr

    log_dir = pathlib.Path("run/logs")
    log_dir.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        filename=log_dir / f'modelbench-{datetime.now().strftime("%y%m%d-%H%M%S")}.log', level=logging.DEBUG
    )
    write_default_config()
    load_plugins(disable_progress_bar=True)


@cli.result_callback()
def at_end(result, **kwargs):
    PROMETHEUS.push_metrics()


@cli.command(help="List known suts")
@local_plugin_dir_option
def list_suts():
    print(SUTS.compact_uid_list())


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
@click.option(
    "sut_uid",
    "--sut",
    "-s",
    multiple=False,
    help="SUT UID to run",
    required=True,
)
@click.option("--anonymize", type=int, help="Randon number seed for consistent anonymization SUTs")
@click.option("--threads", default=32, help="How many threads to use per stage")
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
    type=click.Choice(LOCALES, case_sensitive=False),
    default=DEFAULT_LOCALE,
    help=f"Locale for v1.0 benchmark (Default: {DEFAULT_LOCALE})",
    multiple=False,
)
@click.option(
    "--prompt-set",
    type=click.Choice(list(PROMPT_SETS.keys())),
    default="demo",
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
    sut_uid: str,
    anonymize=None,
    threads=32,
    prompt_set="demo",
    evaluator="default",
) -> None:
    start_time = datetime.now(timezone.utc)
    if locale == "all":
        locales = LOCALES
    else:
        locales = [
            locale.lower(),
        ]

    the_sut = make_sut(sut_uid)

    # benchmark(s)
    benchmarks = [get_benchmark(version, l, prompt_set, evaluator) for l in locales]
    run = run_benchmarks_for_sut(
        benchmarks, the_sut, max_instances, debug=debug, json_logs=json_logs, thread_count=threads
    )
    benchmark_scores = score_benchmarks(run)
    output_dir.mkdir(exist_ok=True, parents=True)
    for b in benchmarks:
        print_summary(b, benchmark_scores, anonymize)
        json_path = output_dir / f"benchmark_record-{b.uid}.json"
        scores = [score for score in benchmark_scores if score.benchmark_definition == b]
        dump_json(json_path, start_time, b, scores)
        print(f"Wrote record for {b.uid} to {json_path}.")
        run_consistency_check(run.journal_path, verbose=True)


@cli.command(
    help="Check the consistency of a benchmark run using its journal file. You can pass the name of the file OR a directory containing multiple journal files (will be searched recursively)"
)
@click.argument("journal-path", type=click.Path(exists=True, dir_okay=True, path_type=pathlib.Path))
# @click.option("--record-path", "-r", type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path))
@click.option("--verbose", "-v", default=False, is_flag=True, help="Print details about the failed checks.")
def consistency_check(journal_path, verbose):
    run_consistency_check(journal_path, verbose)


def run_consistency_check(journal_path, verbose):
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


def ensure_ensemble_annotators_loaded():
    """Check that user has access to the ensemble annotator."""
    try:
        from modelgauge.private_ensemble_annotator_set import EnsembleAnnotatorSet

        return True
    except Exception as e:
        logging.warning(f"Can't load private ensemble annotators: {e}")
        return False


def get_benchmark(version: str, locale: str, prompt_set: str, evaluator: str = "default") -> BenchmarkDefinition:
    """Checks that user has all required secrets and performs basic input validation. Returns a benchmark."""
    assert version == "1.0", ValueError(f"Version {version} is not supported.")
    validate_locale(locale)
    validate_prompt_set(prompt_set, locale)
    if evaluator == "ensemble":
        if not ensure_ensemble_annotators_loaded():
            print(f"Can't build benchmark for {str} {locale} {prompt_set} {evaluator}; couldn't load evaluator.")
            exit(1)

    benchmark = GeneralPurposeAiChatBenchmarkV1(locale, prompt_set, evaluator)
    # Check secrets.
    test_uids = []
    for hazard in benchmark.hazards():
        # Ensure the reference standard exists
        hazard.reference_standard()
        test_uids.extend(hazard.test_uids())
    secrets = load_secrets_from_config()
    check_secrets(secrets, test_uids=test_uids)
    return benchmark


def score_benchmarks(run):
    benchmark_scores = []
    for _, score_dict in run.benchmark_scores.items():
        for _, score in score_dict.items():
            benchmark_scores.append(score)
    return benchmark_scores


def run_benchmarks_for_sut(benchmarks, sut, max_instances, debug=False, json_logs=False, thread_count=32):
    runner = BenchmarkRunner(pathlib.Path("./run"))
    runner.secrets = load_secrets_from_config()
    runner.benchmarks = benchmarks
    runner.sut = sut
    runner.max_items = max_instances
    runner.debug = debug
    runner.thread_count = thread_count
    runner.run_tracker = JsonRunTracker() if json_logs else TqdmRunTracker(0.5)

    print(f"Starting run for {[b.uid for b in benchmarks]} for {sut.uid}")

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


@cli.command(help="Show and optionally update the benchmark standards")
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
    benchmarks = []
    for locale in PUBLISHED_LOCALES:
        for prompt_set in PROMPT_SETS.keys():
            # we do not want to make demo standards. Instead we want to use the practice standards
            if not prompt_set == "demo":
                benchmarks.append(get_benchmark("1.0", locale, prompt_set, "ensemble"))

    reference_sut_uids = ["gemma-2-9b-it-hf", "llama-3.1-8b-instruct-turbo-together"]
    for sut_uid in reference_sut_uids:
        ref_sut = make_sut(sut_uid)
        run_result = run_benchmarks_for_sut(benchmarks, ref_sut, None)
        all_hazard_numeric_scores = defaultdict(list)
        for _, scores_by_sut in run_result.benchmark_scores.items():
            for _, benchmark_score in scores_by_sut.items():
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
            "reference_suts": reference_sut_uids,
            "reference_standards": reference_standards,
        },
    }
    with open(standards_file, "w") as out:
        json.dump(result, out, indent=4)


if __name__ == "__main__":
    cli()
