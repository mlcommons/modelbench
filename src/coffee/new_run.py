import logging
import pathlib
from collections import defaultdict
from typing import Dict, List

import click
from newhelm.base_test import BaseTest
from newhelm.benchmark import BaseBenchmark, Score
from newhelm.general import get_or_create_json_file
from newhelm.placeholders import Result
from newhelm.runners.simple_benchmark_runner import SimpleBenchmarkRunner, run_prompt_response_test
from newhelm.sut_registry import SUTS
from termcolor import termcolor

from coffee.benchmark import GeneralChatBotBenchmarkDefinition, BiasHarmDefinition, BenchmarkScore
from coffee.helm_runner import NewhelmSut
from coffee.static_site_generator import StaticSiteGenerator


def _make_output_dir():
    o = pathlib.Path.cwd()
    if o.name in ["src", "test"]:
        logging.warning(f"Output directory of {o} looks suspicious")
    if not o.name == "run":
        o = o / "run"
    o.mkdir(exist_ok=True)
    return o
import newhelm.tests.bbq
import newhelm.tests.real_toxicity_prompts
class SketchyBenchmark(BaseBenchmark):
    def get_tests(self) -> List[BaseTest]:
        return [newhelm.tests.bbq.BBQ(c) for c in newhelm.tests.bbq._CATEGORIES]
        # return [newhelm.tests.bbq.BBQ(),
        #         # newhelm.tests.real_toxicity_prompts.RealToxicityPrompts(),
        #         ]

    def summarize(self, results: Dict[str, List[Result]]) -> Score:
        bbq = results["BBQ"]
        # count = 0
        # total = 0
        # for subject in bbq:
        #     count += 1
        #     total += bbq[subject]["bbq_accuracy"]
        # return HarmScore(self, total / count)

        result_we_care_about = next(x for x in bbq if x.name == 'accuracy')
        print(f"I should give some scores for {results} ")
        return Score(value=result_we_care_about.value)

@click.command()
@click.option(
    "--output-dir", "-o", default="./web", type=click.Path(file_okay=False, dir_okay=True, path_type=pathlib.Path)
)
@click.option("--max-instances", "-m", type=int, default=100)
@click.option("--debug", default=False, is_flag=True)
@click.option("--web-only", default=False, is_flag=True)
def cli(output_dir: pathlib.Path, max_instances: int, debug: bool, web_only) -> None:
    import newhelm.load_plugins
    newhelm.load_plugins.load_plugins()
    from  newhelm.secrets_registry import SECRETS
    SECRETS.set_values(get_or_create_json_file('secrets/default.json'))
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
                # if not isinstance(harm, BiasHarmDefinition):
                #     print(termcolor.colored(f"skipping {harm} for now"))
                #     continue
                print(termcolor.colored(f'    Examining harm "{harm.name()}"', "yellow"))

                if web_only:
                    # TODO
                    # this is a little sketchy for now, a quick fix to make testing HTML changes easier
                    #tests = itertools.chain(*[harm.tests() for harm in benchmark_definition.harms()])
                    #result = HelmResult(list(tests), suts, pathlib.Path("./run"), None)
                    raise NotImplementedError
                else:
                    tests = harm.tests()
                    # tests = [newhelm.tests.bbq.BBQ(subject=category) for category in newhelm.tests.bbq._CATEGORIES]
                    for test in tests:
                        results[test] = run_prompt_response_test(test, SUTS.make_instance(sut.key), "./run", 5)


                    # relevant = {(key[0],key[1]): results[key] for key in results.keys() if key[2] == sut}
                    score = harm.score(results)
                    if debug:
                        print(
                            termcolor.colored(
                                f"    For harm {harm.name()}, {sut.name} scores {score.value()}",
                                "yellow"
                            )
                        )
                    harm_scores.append(score)
        benchmark_scores.append(BenchmarkScore(benchmark_definition, sut, harm_scores))
    print(benchmark_scores)
        # for sut in suts:
        #     benchmark_scores.append(BenchmarkScore(benchmark_definition, sut, harm_scores_by_sut[sut]))

    print()
    print(termcolor.colored(f"Benchmarking complete, rendering reports...", "green"))
    static_site_generator = StaticSiteGenerator()
    static_site_generator.generate(benchmark_scores, output_dir)
    print()
    print(termcolor.colored(f"Reports complete, open {output_dir}/index.html", "green"))


def run_some_tests():
    results = {}
    for category in newhelm.tests.bbq._CATEGORIES:
        sut = 'gpt2'
        r = run_prompt_response_test(newhelm.tests.bbq.BBQ(subject=category), SUTS.make_instance(sut), "./run", 5)
        results[('BBQ', category, sut)] = r
    print(results)


def run_a_newhelm_benchmark(max_instances):
    runner = SimpleBenchmarkRunner(str(_make_output_dir()), max_test_items=max_instances)
    result = runner.run(SketchyBenchmark(), [SUTS.make_instance('gpt2')])
    print(result)
    print(f"result: {result[0].benchmark_name} {result[0].score}")


if __name__ == "__main__":
    cli()
