import datetime
import faulthandler
import io
import json
import logging
import pathlib
import pkgutil
import signal
import sys
from collections import defaultdict
from datetime import datetime, timezone
from functools import wraps

import click
import termcolor
from click import echo
from rich.console import Console
from rich.table import Table

from modelgauge.config import load_secrets_from_config, write_default_config
from modelgauge.load_namespaces import load_namespaces
from modelgauge.locales import DEFAULT_LOCALE, LOCALES
from modelgauge.monitoring import PROMETHEUS
from modelgauge.preflight import check_secrets, make_sut
from modelgauge.prompt_sets import PROMPT_SETS
from modelgauge.sut import get_sut_and_options
from modelgauge.sut_registry import SUTS

from modelbench.benchmark_runner import BenchmarkRunner, JsonRunTracker, TqdmRunTracker
from modelbench.benchmarks import GeneralPurposeAiChatBenchmarkV1, SecurityBenchmark
from modelbench.consistency_checker import ConsistencyChecker, summarize_consistency_check_results
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


def benchmark_options(func):
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
    @local_plugin_dir_option
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


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
    load_namespaces(disable_progress_bar=True)


@cli.group()
def benchmark() -> None:
    pass


@cli.result_callback()
def at_end(result, **kwargs):
    PROMETHEUS.push_metrics()


@cli.command(help="List known suts")
@local_plugin_dir_option
def list_suts():
    print(SUTS.compact_uid_list())


@benchmark.command("general", help="run a general purpose AI chat benchmark")
@benchmark_options
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
def general_benchmark(
    output_dir: pathlib.Path,
    max_instances: int,
    debug: bool,
    json_logs: bool,
    sut_uid: str,
    version: str,
    locale: str,
    prompt_set="demo",
    evaluator="default",
) -> None:
    # TODO: move this check inside the benchmark class?
    if evaluator == "ensemble":
        if not ensure_ensemble_annotators_loaded():
            print(f"Can't build benchmark for {sut_uid} {locale} {prompt_set} {evaluator}; couldn't load evaluator.")
            exit(1)

    sut_uid, _ = get_sut_and_options(sut_uid)
    sut = make_sut(sut_uid)
    benchmark = GeneralPurposeAiChatBenchmarkV1(locale, prompt_set, evaluator)
    check_benchmark(benchmark)
    run_and_report_benchmark(benchmark, sut, max_instances, debug, json_logs, output_dir)


@benchmark.command("security", help="run a security benchmark")
@benchmark_options
@click.option(
    "--evaluator",
    type=click.Choice(["default", "ensemble"]),
    default="default",
    help="Which evaluator to use",
    show_default=True,
)
def security_benchmark(
    output_dir: pathlib.Path,
    max_instances: int,
    debug: bool,
    json_logs: bool,
    sut_uid: str,
    evaluator="default",
) -> None:
    # TODO: move this check inside the benchmark class?
    if evaluator == "ensemble":
        if not ensure_ensemble_annotators_loaded():
            print("Can't build security benchmark; couldn't load evaluator.")
            exit(1)

    sut_uid, _ = get_sut_and_options(sut_uid)
    sut = make_sut(sut_uid)
    benchmark = SecurityBenchmark(evaluator=evaluator)
    check_benchmark(benchmark)

    run_and_report_benchmark(benchmark, sut, max_instances, debug, json_logs, output_dir)


def run_and_report_benchmark(benchmark, sut, max_instances, debug, json_logs, output_dir):
    start_time = datetime.now(timezone.utc)
    run = run_benchmarks_for_sut([benchmark], sut, max_instances, debug=debug, json_logs=json_logs)

    benchmark_scores = score_benchmarks(run)
    output_dir.mkdir(exist_ok=True, parents=True)
    print_summary(benchmark, benchmark_scores)
    json_path = output_dir / f"benchmark_record-{benchmark.uid}.json"
    scores = [score for score in benchmark_scores if score.benchmark_definition == benchmark]
    dump_json(json_path, start_time, benchmark, scores)
    print(f"Wrote record for {benchmark.uid} to {json_path}.")
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


def check_benchmark(benchmark):
    """Checks that user has all required secrets and performs basic input validation."""
    # TODO: Maybe all these checks should be done in the benchmark constructor?
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


def print_summary(benchmark, benchmark_scores):
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
    "--benchmark",
    "-b",
    type=click.Choice(["general", "security"]),
)
def calibrate_cli(update: bool, benchmark: str) -> None:
    # TODO: Calibrate individual benchmarks.
    if benchmark == "general":
        benchmark_cls = GeneralPurposeAiChatBenchmarkV1
    elif benchmark == "security":
        benchmark_cls = SecurityBenchmark
    else:
        raise ValueError(f"Unknown benchmark type: {benchmark}. Use 'general' or 'security'.")

    echo(f"current standards for {benchmark} benchmark:")
    echo("-----------------")
    echo(json.dumps(benchmark_cls.standards.data, indent=4))

    if update:
        echo()
        benchmark_cls.calibrate()

        echo("new standards")
        echo("-------------")
        echo(json.dumps(benchmark_cls.standards.data, indent=4))


def calibrate(benchmark):
    standards = benchmark.standards
    standards.assert_can_write()

    sut_scores = {}  # Maps SUT UID to a list of its hazard scores
    for sut_uid in benchmark.reference_suts:
        ref_sut = make_sut(sut_uid)
        run_result = run_benchmarks_for_sut([benchmark], ref_sut, None)
        # TODO: Confirm successful run.
        # run_consistency_check(run_result.journal_path, verbose=False)
        scores = run_result.benchmark_scores[benchmark][ref_sut].hazard_scores
        sut_scores[sut_uid] = scores

    standards.write_standards(sut_scores)


if __name__ == "__main__":
    cli()
