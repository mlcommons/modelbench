import datetime
import faulthandler
import io
import logging
import pathlib
import pkgutil
import signal
import sys
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
from modelgauge.log_config import get_file_logging_handler
from modelgauge.monitoring import PROMETHEUS
from modelgauge.preflight import check_secrets, make_sut
from modelgauge.prompt_sets import GENERAL_PROMPT_SETS, SECURITY_JAILBREAK_PROMPT_SETS
from modelgauge.sut_registry import SUTS

from modelbench.benchmark_runner import BenchmarkRunner, JsonRunTracker, TqdmRunTracker
from modelbench.benchmarks import GeneralPurposeAiChatBenchmarkV1, SecurityBenchmark
from modelbench.standards import Standards
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


def benchmark_options(prompt_sets: dict, default_prompt_set: str):
    def decorator(func):
        @click.option(
            "--output-dir",
            "-o",
            default="./run/records",
            type=click.Path(file_okay=False, dir_okay=True, path_type=pathlib.Path),
        )
        @click.option("--max-instances", "-m", type=int, default=None)
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
            type=click.Choice(list(prompt_sets.keys())),
            default=default_prompt_set,
            help="Which prompt set to use",
            show_default=True,
        )
        @click.option(
            "--evaluator",
            type=click.Choice(["default", "official"]),
            default="default",
            help="Which evaluator to use",
            show_default=True,
        )
        @local_plugin_dir_option
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


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
    filename = log_dir / f'modelbench-{datetime.now().strftime("%y%m%d-%H%M%S")}.log'
    logging.basicConfig(level=logging.DEBUG, handlers=[get_file_logging_handler(filename)], force=True)
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
@click.option(
    "--version",
    "-v",
    type=click.Choice(["1.0"]),
    default="1.0",
    help="Benchmark version to run (Default: 1.0)",
    multiple=False,
)
@benchmark_options(GENERAL_PROMPT_SETS, "demo")
def general_benchmark(
    version: str,
    output_dir: pathlib.Path,
    max_instances: int | None,
    debug: bool,
    json_logs: bool,
    sut_uid: str,
    locale: str,
    prompt_set="demo",
    evaluator="default",
) -> None:
    # TODO: move this check inside the benchmark class?
    if evaluator == "official":
        if not ensure_ensemble_annotators_loaded():
            print(f"Can't build benchmark for {sut_uid} {locale} {prompt_set} {evaluator}; couldn't load evaluator.")
            exit(1)
    sut = make_sut(sut_uid)
    benchmark = GeneralPurposeAiChatBenchmarkV1(locale, prompt_set, evaluator)
    check_benchmark(benchmark)
    run_and_report_benchmark(benchmark, sut, max_instances, debug, json_logs, output_dir)


@benchmark.command("security", help="run a security benchmark")
@benchmark_options(SECURITY_JAILBREAK_PROMPT_SETS, "official")
def security_benchmark(
    output_dir: pathlib.Path,
    max_instances: int | None,
    debug: bool,
    json_logs: bool,
    sut_uid: str,
    locale: str,
    prompt_set="official",
    evaluator="default",
) -> None:
    # TODO: move this check inside the benchmark class?
    if evaluator == "official":
        if not ensure_ensemble_annotators_loaded():
            print("Can't build security benchmark; couldn't load evaluator.")
            exit(1)

    sut = make_sut(sut_uid)
    benchmark = SecurityBenchmark(locale, prompt_set, evaluator=evaluator)
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


def run_consistency_check(journal_path, verbose, calibration=False) -> bool:
    """Return True if all checks passed successfully"""
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
    all_passed = True
    for p in journal_paths:
        echo(termcolor.colored(f"\nChecking consistency of journal {p} ..........", "green"))
        try:
            checker = ConsistencyChecker(p, calibration=calibration)
            checker.run(verbose)
            checkers.append(checker)
            if not checker.checks_all_passed():
                all_passed = False
        except Exception as e:
            print("Error running consistency check", e)
            checking_error_journals.append(p)
            all_passed = False

    # Summarize results and unsuccessful checks.
    if len(checkers) > 1:
        echo(termcolor.colored("\nSummary of consistency checks for all journals:", "green"))
        summarize_consistency_check_results(checkers)
    if len(checking_error_journals) > 0:
        echo(termcolor.colored(f"\nCould not run checks on the following journals:", "red"))
        for j in checking_error_journals:
            print("\t", j)
    return all_passed


def ensure_ensemble_annotators_loaded():
    """Check that user has access to the ensemble annotator."""
    try:
        import modelgauge.private_ensemble_annotator_uids

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


def run_benchmarks_for_sut(
    benchmarks,
    sut,
    max_instances,
    debug=False,
    json_logs=False,
    thread_count=32,
    calibrating=False,
    run_path: str = "./run",
):
    runner = BenchmarkRunner(pathlib.Path(run_path), calibrating=calibrating)
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


@cli.command("calibrate", help="Calibrate the benchmark standards")
@click.argument(
    "benchmark_type",
    type=click.Choice(["general", "security"]),
    required=True,
)
@click.option(
    "--locale",
    type=click.Choice(LOCALES, case_sensitive=False),
    required=True,
)
@click.option(
    "--prompt-set",
    required=True,
)
@click.option(
    "--evaluator",
    type=click.Choice(["default", "official"]),
    help="Which evaluator to use",
    show_default=True,
    required=True,
)
def calibrate_cli(benchmark_type: str, locale: str, prompt_set: str, evaluator: str) -> None:
    if benchmark_type == "general":
        benchmark = GeneralPurposeAiChatBenchmarkV1(locale, prompt_set, evaluator=evaluator)
    elif benchmark_type == "security":
        benchmark = SecurityBenchmark(locale, prompt_set, evaluator=evaluator)
    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}. Use 'general' or 'security'.")

    calibrate(benchmark)

    echo("new standards")
    echo("-------------")
    echo(benchmark.standards.dump_data())


def calibrate(benchmark, run_path: str = "./run"):
    reference_benchmark = benchmark.reference_benchmark()
    Standards.assert_can_calibrate_benchmark(reference_benchmark)
    sut_runs = {}
    for sut_uid in reference_benchmark.reference_suts:
        ref_sut = make_sut(sut_uid)
        run_result = run_benchmarks_for_sut([reference_benchmark], ref_sut, None, calibrating=True, run_path=run_path)
        if not run_consistency_check(run_result.journal_path, verbose=True, calibration=True):
            raise RuntimeError(f"Consistency check failed for reference SUT {sut_uid}. Standards not updated.")
        sut_runs[ref_sut] = run_result

    standards = Standards.from_runs(reference_benchmark, sut_runs)
    standards.write()


if __name__ == "__main__":
    cli()
