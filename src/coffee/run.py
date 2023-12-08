import json
import math
import pathlib
import re
import shutil
import subprocess
from abc import abstractmethod, ABC
from collections import defaultdict
from enum import Enum
from typing import List, Tuple

import jq
from jinja2 import Environment, PackageLoader, select_autoescape


# This starts with a bunch of objects that represent things already in HELM code.
# As we shift HELM to accommodate a library use case, it would be nice to compose
# a run directly out of objects/enums/constants, or at least compose RunSpecs from
# exposed pieces that are closely related. E.g., the BbqScenario should know "bbq".


class HelmSut(Enum):
    GPT2 = "huggingface/gpt2"


class HelmTest(ABC):
    # I would like this to be another enum, but BBQ's structural chaos means
    # for now we need custom behavior
    def __init__(self, prefix):
        super().__init__()
        self.prefix = prefix

    @abstractmethod
    def runspecs(self) -> List[str]:
        pass


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
        "Sexual_orientation",
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
        result = defaultdict(dict)
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
        # TODO: make sure the execution succeeded

    def load_scores(self):
        focus = self.output_dir / "benchmark_output" / "runs" / "v1"
        result = HelmScores()
        for t in self.tests:
            for s in self.suts:
                # long term we'll need a lot more data; this is just enough to compute simple scores
                test_sut_scores = {}
                for d in focus.glob(
                    f"{self._filesystem_safe(t.prefix)}:*model={self._filesystem_safe(s.value)}*"
                ):
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

    def _filesystem_safe(self, s: str):
        # reproducing some behavior in HELM; would be nice to remove duplication
        return re.sub("/", "_", s)


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
                    runspecs.append(r + ",model=" + s.value)

        command = self._helm_command_for_runspecs(runspecs, max_instances)

        output_dir = self._make_output_dir()
        execute_result = self._execute(command, output_dir)
        return HelmResult(tests, suts, output_dir, execute_result)

    def _execute(self, command, output_dir):
        return subprocess.run(
            " ".join(command), shell=True, capture_output=True, cwd=output_dir
        )

    def _make_output_dir(self):
        o = pathlib.Path.cwd()
        if o.name in ["src", "test"]:
            o = o.parent
        if not o.name == "run":
            o = o / "run"
        o.mkdir(exist_ok=True)
        return o

    def _helm_command_for_runspecs(self, bbq_runspecs, max_instances):
        command = ["helm-run"]
        command.extend(
            ["--suite", "v1"]
        )  # this is fixed for now, which is probably wrong
        command.extend(["-n", "1"])  # working around a bug
        command.extend(["--max-eval-instances", str(max_instances)])

        command.append("-r")
        command.extend(bbq_runspecs)
        return command


class Benchmark(ABC):
    def __init__(self, sut, scores):
        super().__init__()
        self.sut = sut
        self.scores = scores

    @abstractmethod
    def overall_score(self) -> float:
        pass


class RidiculousBenchmark(Benchmark):
    def overall_score(self) -> float:
        bbq = self.scores["BbqHelmTest"]
        count = 0
        total = 0
        for subject in bbq:
            count += 1
            total += bbq[subject]["bbq_accuracy"]
        return total / count * 5


def quantize_stars(raw_score):
    return round(2 * raw_score) / 2.0


class StaticSiteGenerator:
    def __init__(self) -> None:
        self.env = Environment(
            loader=PackageLoader("src.coffee"), autoescape=select_autoescape()
        )

    # todo: Dedupe this, I mostly just stole it from CliHelmRunner.
    def _make_output_dir(self) -> pathlib.Path:
        o = pathlib.Path.cwd()
        if o.name in ["src", "test"]:
            o = o.parent
        if not o.name == "web":
            o = o / "web"
        if o.exists():
            shutil.rmtree(o, ignore_errors=True)
        o.mkdir(exist_ok=True)
        return o

    def calculate_stars(self, benchmark: Benchmark) -> Tuple[int, bool, int]:
        d, i = math.modf(benchmark.overall_score())
        stars = int(i)
        half_star = d >= 0.5
        empty_stars = 5 - (stars + int(half_star))
        return stars, half_star, empty_stars

    def generate(self, benchmarks: list[Benchmark]) -> None:
        output_dir = self._make_output_dir()
        template = self.env.get_template("benchmark.html")

        for benchmark in benchmarks:
            stars, half_star, empty_stars = self.calculate_stars(benchmark)
            with open(
                pathlib.Path(output_dir, f"{benchmark.sut.name.lower()}.html"), "w+"
            ) as f:
                f.write(
                    template.render(
                        stars=stars,
                        half_star=half_star,
                        empty_stars=empty_stars,
                        benchmark=benchmark,
                    )
                )


if __name__ == "__main__":
    runner = CliHelmRunner()
    suts = [HelmSut.GPT2]
    result = runner.run([BbqHelmTest()], suts, max_instances=100)
    scores = result.load_scores()
    benchmarks = []
    for sut in suts:
        benchmark = RidiculousBenchmark(sut, scores.for_sut(sut))
        benchmarks.append(benchmark)
        print(
            f"{benchmark.sut.name} scored {quantize_stars(benchmark.overall_score())} stars"
        )

    static_site_generator = StaticSiteGenerator()
    static_site_generator.generate(benchmarks)
