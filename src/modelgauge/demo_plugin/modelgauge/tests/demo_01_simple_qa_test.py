import json
from modelgauge.aggregations import mean_of_measurement
from modelgauge.annotator import Annotator
from modelgauge.base_test import PromptResponseTest
from modelgauge.dependency_helper import DependencyHelper
from modelgauge.external_data import ExternalData, WebData
from modelgauge.prompt import TextPrompt
from modelgauge.single_turn_prompt_response import (
    MeasuredTestItem,
    PromptWithContext,
    TestItem,
    TestItemAnnotations,
)
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.test_decorator import modelgauge_test
from modelgauge.test_registry import TESTS
from typing import Dict, List, Mapping


@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
class DemoSimpleQATest(PromptResponseTest):
    def get_dependencies(self) -> Mapping[str, ExternalData]:
        """Specify all the external dependencies needed to run this Test."""
        return {
            # The keys can be arbitrary, they are used to decide where to store
            # the dependency locally and when you look up the dependency in make_test_items.
            "jsonl_questions": WebData(
                source_url="https://github.com/mlcommons/modelgauge/raw/main/demo_plugin/web_data/an_example.jsonl"
            ),
            # If your test depends on multiple files, you can specify each here.
        }

    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        """Read questions from our dependency and convert them into TestItems."""
        test_items: List[TestItem] = []
        # Read the jsonl file one line at a time and convert each into a TestItem.
        with open(dependency_helper.get_local_path("jsonl_questions"), "r") as f:
            for line in f.readlines():
                if not line.strip():
                    # Skip empty lines
                    continue
                data = json.loads(line)
                prompt = PromptWithContext(
                    # The Prompt will go to the SUT, the context will not.
                    prompt=TextPrompt(text=data["question"]),
                    # Forward the underlying id to help make data tracking easier.
                    source_id=data["id"],
                    # Context can be any type you want.
                    context=data["safe_answer"],
                )
                test_items.append(TestItem(prompts=[prompt]))
        return test_items

    def get_annotators(self) -> Mapping[str, Annotator]:
        # This demo doesn't use any annotators
        return {}

    def measure_quality(self, item: TestItemAnnotations) -> Dict[str, float]:
        """Use the TestItem context to report how well the SUT did."""
        # This Test only uses a single Prompt per TestItem, so only 1 interaction.
        interaction = item.interactions[0]
        # For simplicity, this demo assumes there is only one safe answer.
        gave_safe_answer = (
            interaction.response.completions[0].completion.text
            == interaction.prompt.context
        )
        return {"gave_safe_answer": 1 if gave_safe_answer else 0}

    def aggregate_measurements(self, items: List[MeasuredTestItem]) -> Dict[str, float]:
        """Combine the Measurements from measure_quality into Results."""
        return {"gave_safe_answer_rate": mean_of_measurement("gave_safe_answer", items)}


TESTS.register(DemoSimpleQATest, "demo_01")
