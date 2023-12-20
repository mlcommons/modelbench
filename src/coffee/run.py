import dataclasses
import json
import logging
import pathlib
import re
import subprocess
from abc import abstractmethod, ABC
from collections import defaultdict
from enum import Enum
from typing import List

import click
import jq

import coffee
from coffee.benchmark import Benchmark, RidiculousBenchmark
from coffee.static_site_generator import StaticSiteGenerator


# This starts with a bunch of objects that represent things already in HELM code.
# As we shift HELM to accommodate a library use case, it would be nice to compose
# a run directly out of objects/enums/constants, or at least compose RunSpecs from
# exposed pieces that are closely related. E.g., the BbqScenario should know "bbq".
@dataclasses.dataclass
class SutDescription:
    key: str
    huggingface: bool = dataclasses.field(repr=False, default=False)

    def __hash__(self):
        return super().__hash__()


class HelmSut(SutDescription, Enum):
    GPT2 = "openai/gpt2"
    PYTHIA_1B = "EleutherAI/pythia-1b-v0", True
    FB_OPT_125M = "facebook/opt-125m", True


class HelmTest(ABC):
    # I would like this to be another enum, but BBQ's structural chaos means
    # for now we need custom behavior
    def __init__(self, prefix):
        super().__init__()
        self.prefix = prefix

    @abstractmethod
    def runspecs(self) -> List[str]:
        pass

    def __str__(self):
        return self.__class__.__name__ + ":" + self.prefix


class BbqHelmTest(HelmTest):
    def __init__(self):
        super().__init__("bbq")

    # Copied from HELM because it's not exposed outside the method
    CATEGORIES = [
        "Age",
        "Disability_status",
        "Gender_identity",
        "Nationality",
        "Physical_appearance",
        "Race_ethnicity",
        "Race_x_SES",
        "Race_x_gender",
        "Religion",
        "SES",
        # "Sexual_orientation", TODO: temporarily disabled while Yifan looks into a transformer-related bug
    ]

    def runspecs(self) -> List[str]:
        return [f"{self.prefix}:subject={c}" for c in BbqHelmTest.CATEGORIES]


class HelmScores:
    # a kinda hacky container; we won't know the right shape of this for a while, so just use wild dicts
    def __init__(self):
        super().__init__()
        self.data = defaultdict(list)

    def add(self, test, sut, test_sut_scores):
        self.data[(test.__class__.__name__, sut)].append(test_sut_scores)

    def for_sut(self, desired_sut) -> dict:
        result: defaultdict = defaultdict(dict)
        for test, sut in self.data:
            if sut == desired_sut:
                for entry in self.data[(test, sut)]:
                    result[test].update(entry)

        return result


class HelmResult:
    def __init__(
        self,
        tests: List[HelmTest],
        suts: List[HelmSut],
        output_dir: pathlib.Path,
        execution_result: subprocess.CompletedProcess,
    ):
        super().__init__()
        self.tests = tests
        self.suts = suts
        self.output_dir = output_dir
        self.execution_result = execution_result

    def load_scores(self):
        focus = self.output_dir / "benchmark_output" / "runs" / "v1"
        result = HelmScores()
        for t in self.tests:
            for s in self.suts:
                # long term we'll need a lot more data; this is just enough to compute simple scores
                test_sut_scores = {}
                glob_path = f"{self._filesystem_safe(t.prefix)}:*model={self._filesystem_safe(s.key)}*"
                logging.debug(f"looking for scores for {t} {s} in {focus}/{glob_path}")
                for d in focus.glob(glob_path):
                    subject_result = {}
                    with open(d / "run_spec.json") as f:
                        j = json.load(f)
                    subject = (
                        jq.compile(".scenario_spec.args.subject").input_value(j).first()
                    )
                    with open(d / "stats.json") as f:
                        j = json.load(f)
                    for stat in j:
                        if stat["name"]["name"].startswith("bbq_"):
                            subject_result[stat["name"]["name"]] = stat["sum"]
                    test_sut_scores[subject] = subject_result
                result.add(t, s, test_sut_scores)
        return result

    def helm_stdout(self) -> str:
        return self._deal_with_bytes(self.execution_result.stdout)

    def helm_stderr(self) -> str:
        return self._deal_with_bytes(self.execution_result.stderr)

    def _deal_with_bytes(self, o):
        if isinstance(o, bytes):
            result = o.decode("utf-8")
        else:
            result = str(o)
        return result

    def _filesystem_safe(self, s: str):
        # reproducing some behavior in HELM; would be nice to remove duplication
        return re.sub("/", "_", s)

    def success(self):
        return self.execution_result and self.execution_result.returncode == 0


class HelmRunner(ABC):
    @abstractmethod
    def run(self, tests: List[HelmTest], models: List[HelmSut], max_instances=10):
        pass


class CliHelmRunner(HelmRunner):
    def run(self, tests: List[HelmTest], suts: List[HelmSut], max_instances=10):
        runspecs = []
        for s in suts:
            for t in tests:
                for r in t.runspecs():
                    runspecs.append(r + ",model=" + s.key)
        huggingface_models = [s.key for s in suts if s.huggingface]

        command = self._helm_command_for_runspecs(
            runspecs, max_instances, huggingface_models
        )
        logging.debug(f"helm run command: {command}")

        output_dir = self._make_output_dir()
        execute_result = self._execute(command, output_dir)
        return HelmResult(tests, suts, output_dir, execute_result)

    def _execute(
        self, command: List[str], output_dir: pathlib.Path
    ) -> subprocess.CompletedProcess:
        if coffee.app_config.debug:
            return self._run_with_debug_settings(command, output_dir)
        else:
            return subprocess.run(
                " ".join(command), shell=True, capture_output=True, cwd=output_dir
            )

    def _run_with_debug_settings(self, command, output_dir):
        with subprocess.Popen(
            " ".join(command),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=output_dir,
        ) as sp:
            for line in sp.stdout:
                logging.debug(line.decode().rstrip())
        return subprocess.CompletedProcess(sp.args, sp.returncode, sp.stdout, sp.stderr)

    def _make_output_dir(self):
        o = pathlib.Path.cwd()
        if o.name in ["src", "test"]:
            o = o.parent
        if not o.name == "run":
            o = o / "run"
        o.mkdir(exist_ok=True)
        return o

    def _helm_command_for_runspecs(
        self, bbq_runspecs, max_instances, huggingface_models=None
    ):
        command = ["helm-run"]
        command.extend(
            ["--suite", "v1"]
        )  # this is fixed for now, which is probably wrong
        command.extend(["-n", "1"])  # working around a bug
        if huggingface_models:
            for m in huggingface_models:
                command.extend(["--enable-huggingface-models", m])
        command.extend(["--max-eval-instances", str(max_instances)])

        command.append("-r")
        command.extend(bbq_runspecs)
        return command


def quantize_stars(raw_score):
    return round(2 * raw_score) / 2.0


@click.command()
@click.option(
    "--output-dir",
    "-o",
    default="./web",
    type=click.Path(file_okay=False, dir_okay=True, path_type=pathlib.Path),
)
@click.option("--max-instances", "-m", type=int, default=100)
@click.option("--debug", default=False, is_flag=True)
def cli(output_dir: pathlib.Path, max_instances: int, debug: bool) -> None:
    coffee.app_config.debug = debug

    if coffee.app_config.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    runner = CliHelmRunner()
    suts = [HelmSut.FB_OPT_125M, HelmSut.GPT2]
    result = runner.run([BbqHelmTest()], suts, max_instances=max_instances)
    if not result.success():
        print(
            f"HELM execution failed with return code {result.execution_result.returncode}:"
        )
        print("stdout:")
        print(result.helm_stdout())
        print("stderr:")
        print(result.helm_stderr())
    scores = result.load_scores()
    benchmarks: list[Benchmark] = []
    for sut in suts:
        benchmark = RidiculousBenchmark(sut, scores.for_sut(sut))
        benchmarks.append(benchmark)

    static_site_generator = StaticSiteGenerator()
    static_site_generator.generate(benchmarks, output_dir)


if __name__ == "__main__":
    cli()
