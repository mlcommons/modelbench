import os
from modelgauge.aggregations import mean_of_measurement
from modelgauge.annotator import Annotator
from modelgauge.base_test import PromptResponseTest
from modelgauge.data_packing import TarPacker
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
class DemoUnpackingDependencyTest(PromptResponseTest):
    def get_dependencies(self) -> Mapping[str, ExternalData]:
        """Specify all the external dependencies needed to run this Test."""
        return {
            "questions_tar": WebData(
                source_url="https://github.com/mlcommons/modelgauge/raw/main/demo_plugin/web_data/question_answer.tar.gz",
                # Specify that after downloading, this file needs to be unpacked
                # using the Tar command. Because this is specified, get_local_path
                # will return the path to the directory.
                unpacker=TarPacker(),
            ),
        }

    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        """Construction questions from the unpacked dependency."""
        test_items: List[TestItem] = []
        # The question_answer.tar.gz file unpacks to two files:
        # * questions.txt has each question on its own line
        # * answers.txt has the desired answer to each question on its own line.
        data_dir = dependency_helper.get_local_path("questions_tar")
        with open(os.path.join(data_dir, "questions.txt"), "r") as f:
            questions = f.readlines()
        with open(os.path.join(data_dir, "answers.txt"), "r") as f:
            answers = f.readlines()
        # Put the question and answer lines together to make a TestItem.
        for i in range(len(questions)):
            question = questions[i]
            answer = answers[i]
            if not question.strip() or not answer.strip():
                # Skip empty lines
                continue
            prompt = PromptWithContext(
                prompt=TextPrompt(text=question),
                # There is no identifier in the source data, so just pass None.
                source_id=None,
                context=answer,
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


TESTS.register(DemoUnpackingDependencyTest, "demo_02")
