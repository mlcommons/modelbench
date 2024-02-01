import dataclasses
import json
import logging
import pathlib
import re
import subprocess
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import List

import jq
import yaml

import coffee


# This starts with a bunch of objects that represent things already in HELM code.
# As we shift HELM to accommodate a library use case, it would be nice to compose
# a run directly out of objects/enums/constants, or at least compose RunSpecs from
# exposed pieces that are closely related. E.g., the BbqScenario should know "bbq".


@dataclasses.dataclass
class SutDescription:
    key: str
    huggingface: bool = dataclasses.field(repr=False, default=False)
    tokenizer_name: str = ""
    tokenizer_max_length: int = 0

    def __hash__(self):
        return super().__hash__()


class HelmSut(SutDescription, Enum):
    GPT2 = "openai/gpt2"
    PYTHIA_70M = "EleutherAI/pythia-70m", True, "EleutherAI/gpt-neox-20b", 2048
    FB_OPT_125M = "facebook/opt-125m", True, "facebook/opt-66b", 2048


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


class RealToxicityPromptsHelmTest(HelmTest):
    def __init__(self):
        super().__init__("real_toxicity_prompts")

    def runspecs(self) -> List[str]:
        return ["real_toxicity_prompts"]


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
                glob_path = (
                    f"{self._filesystem_safe(t.prefix)}:*model={self._filesystem_safe(s.key)}*"
                )
                logging.debug(f"looking for scores for {t} {s} in {focus}/{glob_path}")
                if t.__class__ == BbqHelmTest:
                    test_sut_scores = self.hackily_extract_bbq_scores(focus, glob_path)
                elif t.__class__ == RealToxicityPromptsHelmTest:
                    test_sut_scores = self.hackily_extract_toxicity_scores(focus, glob_path)
                else:
                    raise NotImplementedError(f"need to add score extration for {t.__class__}")
                result.add(t, s, test_sut_scores)
        return result

    def hackily_extract_bbq_scores(self, focus, glob_path):
        test_sut_scores = {}
        for d in focus.glob(glob_path):
            subject_result = {}
            with open(d / "run_spec.json") as f:
                j = json.load(f)
            subject = jq.compile(".scenario_spec.args.subject").input_value(j).first()
            with open(d / "stats.json") as f:
                j = json.load(f)
            for stat in j:
                if stat["name"]["name"].startswith("bbq_"):
                    subject_result[stat["name"]["name"]] = stat["sum"]
            test_sut_scores[subject] = subject_result
        return test_sut_scores

    def hackily_extract_toxicity_scores(self, focus, glob_path):
        test_sut_scores = {}
        for d in focus.glob(glob_path):
            with open(d / "stats.json") as f:
                j = json.load(f)
            for stat in j:
                if (
                    stat["name"]["split"] == "test"
                    and stat["name"]["sub_split"] == "toxic"
                    and "perturbation" not in stat["name"]
                ):
                    test_sut_scores[stat["name"]["name"]] = stat["sum"]
        return test_sut_scores

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
        runspecs = self._build_runspecs(suts, tests)

        command = self._helm_command_for_runspecs(runspecs, max_instances)
        logging.debug(f"helm run command: {command}")

        output_dir = self._make_output_dir()
        huggingface_models = [s for s in suts if s.huggingface]

        prod_env_dir = output_dir / "prod_env"
        prod_env_dir.mkdir(exist_ok=True)
        md_conf = [self._model_deployment_conf(m) for m in huggingface_models]
        with open(prod_env_dir / "model_deployments.yaml", "w") as out:
            yaml.dump({"model_deployments": md_conf}, out)

        execute_result = self._execute(command, output_dir)
        return HelmResult(tests, suts, output_dir, execute_result)

    def _build_runspecs(self, suts, tests):
        runspecs = []
        for s in suts:
            for t in tests:
                for r in t.runspecs():
                    if ":" in r:
                        separator = ","
                    else:
                        separator = ":"
                    runspecs.append(r + separator + "model=" + s.key)
        return runspecs

    def _execute(self, command: List[str], output_dir: pathlib.Path) -> subprocess.CompletedProcess:
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

    def _helm_command_for_runspecs(self, bbq_runspecs, max_instances):
        command = [
            "python " + str(pathlib.Path(__file__).parent.parent / "dubious_helm_cli_wrapper.py")
        ]
        command.extend(["--suite", "v1"])  # this is a fixed string for now, which is probably wrong
        command.extend(["-n", "1"])  # working around a bug
        command.extend(["--max-eval-instances", str(max_instances)])

        command.append("-r")
        command.extend(bbq_runspecs)
        return command

    def _model_deployment_conf(self, sut: HelmSut):
        return {
            "name": sut.key,
            "tokenizer_name": sut.tokenizer_name,
            "max_sequence_length": sut.tokenizer_max_length,
            "client_spec": {
                "class_name": "helm.proxy.clients.huggingface_client.HuggingFaceClient"
            },
        }
