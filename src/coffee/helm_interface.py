import subprocess
from typing import Iterable

import helm.benchmark.run_specs
from helm.benchmark.config_registry import (
    register_builtin_configs_from_helm_package,
)
from helm.benchmark.executor import ExecutionSpec
from helm.benchmark.huggingface_registration import (
    register_huggingface_hub_model_from_flag_value,
)
from helm.benchmark.model_deployment_registry import (
    ClientSpec,
    ModelDeployment,
    register_model_deployment,
)
from helm.benchmark.presentation.run_entry import RunEntry
from helm.benchmark.run import run_entries_to_run_specs
from helm.benchmark.runner import Runner
from helm.common.authentication import Authentication

from coffee.helm_runner import HelmResult, HelmRunner, HelmSut, HelmTest

helm.benchmark.run_specs.INCLUDE_GENERATIVE_HARMS_METRICS = True


class InProcessHelmRunner(HelmRunner):
    def run(self, tests: list[HelmTest], suts: list[HelmSut], max_instances=10):
        self._execute(
            tests,
            suts,
            max_eval_instances=max_instances,
            suite="v1",
            num_threads=4,
            benchmark_output_path="run/benchmark_output",
            prod_env_path="run/prod_env",
        )

        output_dir = self._make_output_dir()

        # THIS IS A BIG, DUMB HACK until we unwind subprocess.CompletedProcess from the run mix.
        execution_result = subprocess.run("", shell=True, capture_output=True, cwd=output_dir)
        # END BIG DUMB HACK

        return HelmResult(tests, suts, output_dir, execution_result)

    def _execute(
        self,
        tests: Iterable["HelmTest"],
        suts: Iterable["HelmSut"],
        max_eval_instances: int = 10,
        suite: str = "v1",
        num_threads: int = 1,
        benchmark_output_path: str = "run/benchmark_output",
        prod_env_path: str = "run/prod_env",
    ) -> None:
        register_builtin_configs_from_helm_package()
        for sut in suts:
            if sut.huggingface:
                register_huggingface_hub_model_from_flag_value(sut.key)
                model_deployment = ModelDeployment(
                    name=sut.key,
                    tokenizer_name=sut.tokenizer_name,
                    max_sequence_length=sut.tokenizer_max_length,
                    client_spec=ClientSpec(class_name="helm.proxy.clients.huggingface_client.HuggingFaceClient"),
                )
                register_model_deployment(model_deployment)
        run_entries = [RunEntry(r, 1, list()) for r in self._build_runspecs(suts, tests)]
        run_specs = run_entries_to_run_specs(run_entries, max_eval_instances=max_eval_instances)
        execution_spec = ExecutionSpec(
            url=None,
            auth=Authentication(""),
            local_path=prod_env_path,
            parallelism=num_threads,
        )
        runner = Runner(
            execution_spec=execution_spec,
            output_path=benchmark_output_path,
            suite=suite,
            skip_instances=False,
            cache_instances=False,
            cache_instances_only=False,
            skip_completed_runs=False,
            exit_on_error=False,
        )
        runner.run_all(run_specs)
