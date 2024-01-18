from dataclasses import replace
import os
from typing import List, Mapping
from helm.benchmark.scenarios.scenario import create_scenario
from helm.benchmark.run_specs import get_bbq_spec
from helm.benchmark.data_preprocessor import DataPreprocessor
from helm.benchmark.adaptation.adapters.adapter_factory import AdapterFactory
from helm.benchmark.config_registry import (
    register_builtin_configs_from_helm_package,
)
from helm.benchmark.window_services.tokenizer_service import TokenizerService
from helm.benchmark.window_services.default_window_service import DefaultWindowService
from newhelm.base_test import BasePromptResponseTest, TestMetadata
from newhelm.placeholders import Measurement, Prompt, Result
from newhelm.external_data import ExternalData

from newhelm.single_turn_prompt_response import (
    TestItemAnnotations,
    MeasuredTestItem,
    PromptWithContext,
    TestItem,
)


class FakeTokenizerService(TokenizerService):
    def __init__(self):
        pass


class FakeWindowService(DefaultWindowService):
    def __init__(self):
        pass

    def fits_within_context_window(self, *args, **kwargs):
        return True


class BBQImported(BasePromptResponseTest):
    @classmethod
    def get_metadata(cls) -> TestMetadata:
        """Return a description of the test."""
        return TestMetadata("ImportedBBQ", "BBQ implemented using imports of HELM.")

    @classmethod
    def get_dependencies(cls) -> Mapping[str, ExternalData]:
        """Can't declare the dependencies because they are baked into HELM."""
        return {}

    def __init__(self):
        register_builtin_configs_from_helm_package()
        self.run_spec = get_bbq_spec("all")
        # The model_deployment has to be real in order to pass some helm checks,
        # but otherwise we don't need it.
        adapter_spec = replace(
            self.run_spec.adapter_spec, model_deployment="simple/model1"
        )
        self.run_spec = replace(self.run_spec, adapter_spec=adapter_spec)

    def make_test_items(self, dependency_helper) -> List[TestItem]:
        scenario = create_scenario(self.run_spec.scenario_spec)

        scenario_cache_path = os.path.join(
            "benchmark_output", "scenarios", scenario.name
        )
        instances = scenario.get_instances(scenario_cache_path)
        instances = DataPreprocessor(self.run_spec.data_augmenter_spec).preprocess(
            instances
        )

        adapter = AdapterFactory.get_adapter(
            self.run_spec.adapter_spec, FakeTokenizerService()
        )
        # Monkey patch the window service
        adapter.window_service = FakeWindowService()
        scenario_state = adapter.adapt(instances, 1)

        # This ignores train trials, etc
        test_items = []
        for state in scenario_state.request_states:
            prompt_text = state.request.prompt
            test_items.append(
                TestItem(
                    prompts=[PromptWithContext(Prompt(prompt_text), context=state)]
                )
            )

        return test_items

    def measure_quality(self, item: TestItemAnnotations) -> List[Measurement]:
        """Use the SUT responses with annotations to determine how well the SUT did on this TestItem."""
        # TODO
        return []

    def aggregate_measurements(self, items: List[MeasuredTestItem]) -> List[Result]:
        """Combine the measurements for each TestItem into a list of Results."""
        # TODO
        return []
