from itertools import product
from typing import Iterable, TYPE_CHECKING

from helm.benchmark.config_registry import (
    register_builtin_configs_from_helm_package,
)
from helm.benchmark.executor import ExecutionSpec
from helm.benchmark.huggingface_registration import (
    register_huggingface_hub_model_from_flag_value,
)
from helm.benchmark.presentation.run_entry import RunEntry
from helm.benchmark.run import run_entries_to_run_specs
from helm.benchmark.runner import Runner
from helm.common.authentication import Authentication

if TYPE_CHECKING:
    from helm_runner import HelmSut, HelmTest

from helm.benchmark.runner import RunnerError


def run_executions(
    tests: Iterable["HelmTest"],
    suts: Iterable["HelmSut"],
    max_eval_instances: int = 10,
    suite: str = "v1",
    num_threads: int = 4,
    benchmark_output_path: str = "run/benchmark_output",
    prod_env_path: str = "run/prod_env",
) -> None:
    register_builtin_configs_from_helm_package()
    for sut in suts:
        if sut.huggingface:
            register_huggingface_hub_model_from_flag_value(sut.key)
    run_entries = []
    for test, sut in product(tests, suts):
        for runspec in test.runspecs():
            run_entries.append(
                RunEntry(
                    description=f"{runspec},model={sut.key}", priority=1, groups=[]
                )
            )
    run_specs = run_entries_to_run_specs(
        run_entries, max_eval_instances=max_eval_instances
    )
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
