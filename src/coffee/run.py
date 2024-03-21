import csv
import json
import logging
import os
import pathlib
import platform
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Mapping

import click
import newhelm
import termcolor
from click import echo
from newhelm.config import load_secrets_from_config, write_default_config
from newhelm.instance_factory import FactoryEntry
from newhelm.load_plugins import load_plugins
from newhelm.runners.simple_test_runner import run_prompt_response_test
from newhelm.sut_registry import SUTS
from newhelm.test_registry import TESTS
from retry import retry

from coffee.benchmark import (
    BenchmarkScore,
    GeneralChatBotBenchmarkDefinition,
    GeneralPurposeAiChatModel,
    HazardDefinition,
    HazardScore,
    STANDARDS,
)
from coffee.newhelm_runner import NewhelmSut
from coffee.static_site_generator import StaticSiteGenerator

_DEFAULT_SUTS = NewhelmSut

#     [
#     NewhelmSut.GPT2,
#     NewhelmSut.LLAMA_2_7B,
#     NewhelmSut.LLAMA_2_13B,
#     NewhelmSut.LLAMA_2_70B,
#     NewhelmSut.MISTRAL_7B,
#     NewhelmSut.MISTRAL_7B_INSTRUCT,
#     NewhelmSut.PYTHIA_70M,
#     NewhelmSut.YI_BASE_6B,
# ]


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
@click.option("--view-embed", default=False, is_flag=True, help="Render the HTML to be embedded in another view")
def benchmark(
    output_dir: pathlib.Path, max_instances: int, debug: bool, web_only, sut: List[str], view_embed: bool
) -> None:
    secrets = load_secrets_from_config()
    suts = [s for s in NewhelmSut if s.key in sut]
    benchmark_scores = []
    # benchmarks = [GeneralChatBotBenchmarkDefinition()]
    benchmarks = [GeneralPurposeAiChatModel()]
    for sut in suts:
        echo(termcolor.colored(f'Examining system "{sut.display_name}"', "green"))
        sut_instance = SUTS.make_instance(sut.key, secrets=secrets)
        benchark_start_time = datetime.now(timezone.utc)
        for benchmark_definition in benchmarks:
            echo(termcolor.colored(f'  Starting run for benchmark "{benchmark_definition.name()}"', "green"))
            hazard_scores = []
            for hazard in benchmark_definition.hazards():
                echo(termcolor.colored(f'    Examining hazard "{hazard.name()}"', "green"))

                if web_only:
                    # TODO load result from disk here
                    raise NotImplementedError
                else:
                    results = {}
                    hazard_start_time = datetime.now(timezone.utc)
                    for test in hazard.tests(secrets=secrets):
                        items = max_instances
                        if isinstance(test, newhelm.tests.bbq.BBQ):
                            # BBQ is currently multiple sub-tests, so roughly split the items among them
                            items = int(items / len(newhelm.tests.bbq._CATEGORIES))
                            if items <= 0:
                                # Ensure we always do at least one.
                                items = 1
                        results[test.uid] = run_prompt_response_test(
                            test=test, sut=sut_instance, data_dir="./run", max_test_items=items
                        )

                    hazard_end_time = datetime.now(timezone.utc)
                    score = hazard.score(results, hazard_start_time, hazard_end_time)
                    if debug:
                        echo(
                            termcolor.colored(
                                f"    For hazard {hazard.name()}, {sut.name} scores {score.value()}", "green"
                            )
                        )
                    hazard_scores.append(score)
            benchmark_end_time = datetime.now(timezone.utc)
            benchmark_scores.append(
                BenchmarkScore(benchmark_definition, sut, hazard_scores, benchark_start_time, benchmark_end_time)
            )

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
    secrets = load_secrets_from_config()
    reference_sut = NewhelmSut.PYTHIA_70M
    hazards = GeneralChatBotBenchmarkDefinition(secrets=secrets).hazards()
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
            "3_star": {
                hazard.__class__.name(): hazard_scores[hazard].score.estimate for hazard in hazard_scores.keys()
            },
        },
    }
    with open(file, "w") as out:
        json.dump(result, out, indent=4)


def run_tests(hazards: List[HazardDefinition], sut: NewhelmSut, items: int) -> Mapping[HazardDefinition, HazardScore]:
    secrets = load_secrets_from_config()
    result = {}
    sut_instance = SUTS.make_instance(sut.key, secrets=secrets)
    for hazard in hazards:
        test_scores = {}
        for test_key, test in hazard.tests().items():
            test_scores[test_key] = run_prompt_response_test(test_key, test, sut.key, sut_instance, "./run", items)
        result[hazard] = hazard.score(test_scores)
    return result


def test_records_for_sut(sut: NewhelmSut, tests: Dict[str, FactoryEntry], data_dir="./test_result_generator"):
    secrets = load_secrets_from_config()
    for test_id, factory in tests.items():
        test = factory.make_instance(secrets=secrets)

        @retry(tries=3, delay=1)  # to compensate for newhelm/togetherai flakiness
        def process_one():
            result = run_prompt_response_test(test, SUTS.make_instance(sut.key, secrets=secrets), data_dir)
            return test_id, result

        try:
            yield process_one()
        except Exception as e:
            print(f"Unexpected failure for {sut.name} on {test_id}", file=sys.stderr)
            raise e


# def some_convenient_tests():
#     tests = dict(TESTS.items())
#     for key in list(tests.keys()):
#         if not key.startswith("safe-"):
#             del tests[key]
#     del tests["safe-cbr"]  # currently broken due to blanks in data
#     return tests
#
#
# @cli.command(help="produce a CSV grid of test scores for some convenient tests and SUTs")
# @click.option("--output", type=click.File("w"))
# def grid(output):
#     if not output:
#         output = sys.stdout
#
#     tests = some_convenient_tests()
#     test_item_counts = {}
#
#     out = csv.writer(output)
#     out.writerow(["SUT"] + list(tests.keys()))
#     for sut in _DEFAULT_SUTS:
#         output_row = [sut.key]
#         for test_id, result in test_records_for_sut(sut, tests, "./grid_run"):
#             output_row.append(result.result.data["frac_safe"])
#             test_item_counts[test_id] = len(result.test_item_records)
#         out.writerow(output_row)
#         output.flush()
#     out.writerow([])
#     out.writerow(["counts"] + [test_item_counts[k] for k in tests.keys()])
#
#
# @cli.command(help="produce CSVs of the responses for the grid's tests and suts")
# @click.argument("output", type=click.Path(file_okay=False, path_type=pathlib.Path))
# def results(output: pathlib.Path):
#     def clean_text(str):
#         return re.sub(r"\s+", " ", str.replace("\n", " ")).strip()
#
#     """
#     This is super dependent on the particular results from particular tests. If those change, this will need to change
#     """
#     if not output.exists():
#         output.mkdir(parents=True)
#     test_records = defaultdict(lambda: dict())
#     for sut in _DEFAULT_SUTS:
#         for test_id, test_record in test_records_for_sut(sut, some_convenient_tests(), "./grid_run"):
#             test_records[test_id][sut.key] = test_record
#     for test_id in test_records.keys():
#         with open(output / f"test_items-{test_id}.csv", "w") as test_file:
#             writer = csv.writer(test_file)
#             writer.writerow(["prompt_id", "sut", "is_safe", "violation_categories", "prompt", "response"])
#             rows = []
#             for sut_key, test_record in test_records[test_id].items():
#                 for test_item_record in test_record.test_item_records:
#                     rows.append(
#                         [
#                             test_item_record.test_item.prompts[0].source_id,
#                             sut_key,
#                             test_item_record.measurements["is_safe"],
#                             "|".join(
#                                 test_item_record.annotations["llama_guard"].data["interactions"][0]["completions"][0][
#                                     "violation_categories"
#                                 ]
#                             ),
#                             clean_text(test_item_record.interactions[0].prompt.prompt.text),
#                             clean_text(test_item_record.interactions[0].response.completions[0].text),
#                         ]
#                     )
#
#             for row in sorted(rows, key=lambda r: (r[0], r[1])):
#                 writer.writerow(row)


if __name__ == "__main__":
    cli()
