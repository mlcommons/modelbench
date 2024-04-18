import csv
import functools
import itertools
import json
import os
import pathlib
import platform
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool
from typing import List, Mapping, Dict

import casefy
import click
import termcolor
from click import echo
from modelbench.benchmarks import (
    BenchmarkDefinition,
    GeneralPurposeAiChatBenchmark,
)
from modelbench.hazards import HazardDefinition, HazardScore, STANDARDS
from modelbench.modelgauge_runner import ModelGaugeSut, SutDescription
from modelbench.static_site_generator import StaticSiteGenerator, StaticContent
from modelgauge.config import load_secrets_from_config, write_default_config
from modelgauge.instance_factory import FactoryEntry
from modelgauge.load_plugins import load_plugins
from modelgauge.simple_test_runner import run_prompt_response_test
from modelgauge.sut_registry import SUTS
from modelgauge.test_registry import TESTS
from modelgauge.tests.safe import SafeTestResult
from retry import retry

_DEFAULT_SUTS = ModelGaugeSut


@click.group()
def cli() -> None:
    write_default_config()
    load_plugins()
    print()
    print(StaticContent()["general"]["provisional_disclaimer"])
    print()


@cli.command(help="run the standard benchmark")
@click.option(
    "--output-dir", "-o", default="./web", type=click.Path(file_okay=False, dir_okay=True, path_type=pathlib.Path)
)
@click.option("--max-instances", "-m", type=int, default=100)
@click.option("--debug", default=False, is_flag=True)
@click.option(
    "--sut",
    "-s",
    multiple=True,
)
@click.option("--view-embed", default=False, is_flag=True, help="Render the HTML to be embedded in another view")
@click.option("--anonymize", type=int, help="Random number seed for consistent anonymization of SUTs")
@click.option("--parallel", default=False, help="experimentally run SUTs in parallel")
def benchmark(
    output_dir: pathlib.Path,
    max_instances: int,
    debug: bool,
    sut: List[str],
    view_embed: bool,
    anonymize=None,
    parallel=False,
) -> None:
    suts = find_suts_for_sut_argument(sut)
    benchmarks = [GeneralPurposeAiChatBenchmark()]
    benchmark_scores = score_benchmarks(benchmarks, suts, max_instances, debug, parallel)
    generate_content(benchmark_scores, output_dir, anonymize, view_embed)


def find_suts_for_sut_argument(sut_args: List[str]):
    if sut_args:
        suts = []
        default_suts_by_key = {s.key: s for s in ModelGaugeSut}
        registered_sut_keys = set(i[0] for i in SUTS.items())
        for sut_arg in sut_args:
            if sut_arg in default_suts_by_key:
                suts.append(default_suts_by_key[sut_arg])
            elif sut_arg in registered_sut_keys:
                suts.append(SutDescription(sut_arg, re.sub(r"[-_]+", " ", sut_arg)))
            else:
                all_sut_keys = registered_sut_keys.union(set(default_suts_by_key.keys()))
                raise click.BadParameter(
                    f"Unknown key '{sut_arg}'. Valid options are {sorted(all_sut_keys, key=lambda x:x.lower())}",
                    param_hint="sut",
                )

    else:
        suts = ModelGaugeSut
    return suts


def score_benchmarks(benchmarks, suts, max_instances, debug, parallel=True):
    secrets = load_secrets_from_config()
    if parallel:
        f = functools.partial(score_a_sut, benchmarks, max_instances, secrets, debug)
        with Pool(len(suts)) as p:
            results = p.map(f, suts)
            return list(itertools.chain(*results))
    else:
        benchmark_scores = []
        for sut in suts:
            sut_scores = score_a_sut(benchmarks, max_instances, secrets, debug, sut)
            benchmark_scores.extend(sut_scores)
        return benchmark_scores


def score_a_sut(benchmarks, max_instances, secrets, debug, sut):
    sut_scores = []
    echo(termcolor.colored(f'Examining system "{sut.display_name}"', "green"))
    sut_instance = SUTS.make_instance(sut.key, secrets=secrets)
    for benchmark_definition in benchmarks:
        echo(termcolor.colored(f'  Starting run for benchmark "{benchmark_definition.name()}"', "green"))
        hazard_scores = []
        for hazard in benchmark_definition.hazards():
            echo(termcolor.colored(f'    Examining hazard "{hazard.name()}"', "green"))

            results = {}
            for test in hazard.tests(secrets=secrets):
                items = max_instances
                results[test.uid] = run_prompt_response_test(
                    test=test, sut=sut_instance, data_dir="./run", max_test_items=items
                )

            score = hazard.score(results)
            if debug:
                echo(termcolor.colored(f"    For hazard {hazard.name()}, {sut.name} scores {score.value()}", "green"))
            hazard_scores.append(score)
        benchmark_end_time = datetime.now(timezone.utc)
        sut_scores.append(benchmark_definition.score(sut, hazard_scores, benchmark_end_time))
    return sut_scores


def generate_content(benchmark_scores, output_dir, anonymize, view_embed):
    static_site_generator = StaticSiteGenerator(view_embed=view_embed)
    if anonymize:

        class FakeSut(SutDescription):
            @property
            def name(self):
                return self.key.upper()

        rng = random.Random(anonymize)
        rng.shuffle(benchmark_scores)

        counter = 0
        for bs in benchmark_scores:
            counter += 1
            key = f"sut{counter:02d}"
            name = f"System Under Test {counter}"

            bs.sut = FakeSut(key, name)
            static_site_generator._content[key] = {"name": name, "tagline": "A well-known model."}
    echo()
    echo(termcolor.colored(f"Benchmarking complete, rendering reports...", "green"))
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


def update_standards_to(file):
    reference_suts = [ModelGaugeSut.VICUNA_13B, ModelGaugeSut.MISTRAL_7B, ModelGaugeSut.WIZARDLM_13B]
    hazards = list(itertools.chain.from_iterable([bm().hazards() for bm in BenchmarkDefinition.__subclasses__()]))
    all_results = {h.key(): [] for h in hazards}
    for sut in reference_suts:
        test_results = run_tests(hazards, sut, 9000)
        for d, r in test_results.items():
            all_results[d.key()].append(r.score.estimate)
    reference_standards = {d: min(s) for d, s in all_results.items() if s}
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
            "reference_suts": [{"name": sut.display_name, "id": sut.key} for sut in reference_suts],
            "reference_standards": reference_standards,
        },
    }
    with open(file, "w") as out:
        json.dump(result, out, indent=4)


def run_tests(
    hazards: List[HazardDefinition], sut: ModelGaugeSut, items: int
) -> Mapping[HazardDefinition, HazardScore]:
    secrets = load_secrets_from_config()
    result = {}
    sut_instance = SUTS.make_instance(sut.key, secrets=secrets)
    for hazard in hazards:
        test_scores = {}
        for test in hazard.tests(secrets=secrets):
            test_scores[test.uid] = run_prompt_response_test(
                test=test, sut=sut_instance, data_dir="./run", max_test_items=items
            )
        result[hazard] = hazard.score(test_scores)
    return result


def test_records_for_sut(sut: ModelGaugeSut, tests: Dict[str, FactoryEntry], data_dir="./run", max_test_items=100):
    secrets = load_secrets_from_config()
    for test_id, factory in tests.items():
        test = factory.make_instance(secrets=secrets)

        @retry(tries=3, delay=1)  # to compensate for modelgauge/togetherai flakiness
        def process_one():
            result = run_prompt_response_test(
                test, SUTS.make_instance(sut.key, secrets=secrets), data_dir, max_test_items=max_test_items
            )
            return test_id, result

        try:
            yield process_one()
        except Exception as e:
            print(f"Unexpected failure for {sut.name} on {test_id}", file=sys.stderr)
            raise e


def some_convenient_tests():
    tests = dict(TESTS.items())
    for key in list(tests.keys()):
        if not key.startswith("safe-"):
            del tests[key]
        if key == "safe-ben":
            del tests[key]
    return tests


@cli.command(help="produce a CSV grid of test scores for some convenient tests and SUTs")
@click.option("--output", type=click.File("w"))
@click.option("--max-instances", "-m", type=int, default=100)
def grid(output, max_instances: int) -> None:
    if not output:
        output = sys.stdout

    tests = some_convenient_tests()
    test_item_counts = {}

    out = csv.writer(output)
    out.writerow(["SUT"] + list(tests.keys()))
    for sut in _DEFAULT_SUTS:
        output_row = [sut.key]
        for test_id, test_record in test_records_for_sut(sut, tests, "./run", max_test_items=max_instances):
            result = test_record.result.to_instance(SafeTestResult)
            safe_count = 0
            item_count = 0
            for persona, persona_result in result.persona_results.items():
                item_count += persona_result.num_items
                safe_count += persona_result.num_items * persona_result.frac_safe
            output_row.append(safe_count / item_count)
            if test_id in test_item_counts:
                assert test_item_counts[test_id] == item_count
            else:
                test_item_counts[test_id] = item_count
        out.writerow(output_row)
        output.flush()
    out.writerow([])
    out.writerow(["counts"] + [test_item_counts[k] for k in tests.keys()])


@cli.command(help="produce CSVs of the responses for the grid's tests and suts")
@click.argument("output", type=click.Path(file_okay=False, path_type=pathlib.Path))
@click.option("--max-instances", "-m", type=int, default=100)
def responses(output: pathlib.Path, max_instances: int) -> None:
    def clean_text(str):
        return re.sub(r"\s+", " ", str.replace("\n", " ")).strip()

    """
    This is super dependent on the particular results from particular tests. If those change, this will need to change
    """
    if not output.exists():
        output.mkdir(parents=True)
    test_records = defaultdict(lambda: dict())
    for sut in _DEFAULT_SUTS:
        for test_id, test_record in test_records_for_sut(
            sut, some_convenient_tests(), "./run", max_test_items=max_instances
        ):
            test_records[test_id][sut.key] = test_record
    for test_id in test_records.keys():
        with open(output / f"test_items-{test_id}.csv", "w") as test_file:
            writer = csv.writer(test_file)
            writer.writerow(["prompt_id", "sut", "is_safe", "violation_categories", "prompt", "response"])
            rows = []
            for sut_key, test_record in test_records[test_id].items():
                for test_item_record in test_record.test_item_records:
                    rows.append(
                        [
                            test_item_record.test_item.prompts[0].source_id,
                            sut_key,
                            test_item_record.measurements["is_safe"],
                            "|".join(
                                test_item_record.interactions[0]
                                .response.completions[0]
                                .annotations["llama_guard"]
                                .data["violation_categories"]
                            ),
                            clean_text(test_item_record.interactions[0].prompt.prompt.text),
                            clean_text(test_item_record.interactions[0].response.completions[0].completion.text),
                        ]
                    )

            for row in sorted(rows, key=lambda r: (r[0], r[1])):
                writer.writerow(row)


if __name__ == "__main__":
    cli()
