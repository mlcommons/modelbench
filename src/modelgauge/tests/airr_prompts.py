import csv
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Dict, List, Mapping

from modelgauge.base_test import PromptResponseTest
from modelgauge.dependency_helper import DependencyHelper
from modelgauge.external_data import ExternalData, LocalData
from modelgauge.prompt import SUTOptions, TextPrompt
from modelgauge.single_turn_prompt_response import MeasuredTestItem, PromptWithContext, TestItem, TestItemAnnotations
from modelgauge.sut_capabilities import AcceptsTextPrompt, ProducesPerTokenLogProbabilities
from modelgauge.test_decorator import modelgauge_test
from modelgauge.test_registry import TESTS



class AirrPrompts(PromptResponseTest, ABC):
    def __init__(self, uid: str, temp: float):
        super().__init__(uid)
        self.temp = temp
        self.sut_options = SUTOptions(max_tokens=500, temperature=self.temp, top_logprobs=1)

    @abstractmethod
    def _get_prompt_set_names(self):
        pass

    @abstractmethod
    def _make_item(self, row) -> TestItem:
        pass

    def get_dependencies(self) -> Mapping[str, ExternalData]:
        root_path = "/home/barbara/input"
        dependencies = {}
        for prompt_set in self._get_prompt_set_names():
            dependencies[prompt_set] = LocalData(path=f"{root_path}/{prompt_set}.csv")
        return dependencies

    @classmethod
    def get_annotators(cls) -> List[str]:
        return []

    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        test_items: List[TestItem] = []
        for prompt_set in self._get_prompt_set_names():
            data_file = dependency_helper.get_local_path(prompt_set)
            with open(data_file, "r") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    text = row["prompt_text"].strip()
                    if not text:
                        continue
                    test_items.append(self._make_item(row))


        return test_items

    def measure_quality(self, item: TestItemAnnotations) -> Dict[str, float]:
        return {}

    def aggregate_measurements(self, items: List[MeasuredTestItem]):
        return {"num_prompts_successful": len(items)}

PRACTICE_FILE_NAME = "airr_official_1.0_practice_prompt_set"
HELDBACK_FILE_NAME = "airr_official_1.0_heldback_prompt_set_internal_use_only"


class PracticeContext(BaseModel):
    # CSV:  internal_id, prompt_text, hazard, locale, prompt_hash
    hazard: str
    locale: str
    prompt_hash: str


class HeldbackContextInternal(BaseModel):
    # internal_id,prompt_uid,batch_uid,prompt_text,disposition,vendor,number,hazard,hazard_subtype,persona,locale,original_language,original_prompt_uid,technique,source,tags,prompt_hash,creation_time,update_time
    internal_id: str
    prompt_uid: str
    batch_uid: str
    hazard: str
    prompt_hash: str
    locale: str


@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt, ProducesPerTokenLogProbabilities])
class PracticePrompts(AirrPrompts):
    def _get_prompt_set_names(self):
        return [PRACTICE_FILE_NAME]

    def _make_item(self, row) -> TestItem:
        text = row["prompt_text"].strip()
        prompt = PromptWithContext(
            prompt=TextPrompt(text=text, options=self.sut_options),
            source_id=row["internal_id"],
        )
        return TestItem(
                prompts=[prompt],
                context=PracticeContext(
                    hazard=row["hazard"], locale=row["locale"], prompt_hash=row["prompt_hash"]
                ),
            )


@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt, ProducesPerTokenLogProbabilities])
class HeldbackPromptsInternal(AirrPrompts):
    """Uses internal_use_only version."""

    def _get_prompt_set_names(self):
        return [HELDBACK_FILE_NAME]

    def _make_item(self, row) -> TestItem:
        text = row["prompt_text"].strip()
        prompt = PromptWithContext(
            prompt=TextPrompt(text=text, options=self.sut_options),
            source_id=row["internal_id"],
        )
        return TestItem(
                prompts=[prompt],
                context=HeldbackContextInternal(
                   internal_id=row["internal_id"], prompt_uid=row["prompt_uid"], batch_uid=row["batch_uid"], hazard=row["hazard"], prompt_hash=row["prompt_hash"], locale=row["locale"]
                ),
            )



temps = [0.01]
for temp in temps:
    TESTS.register(PracticePrompts, f"{PRACTICE_FILE_NAME}-no_eval-temp_{temp}", temp)
    TESTS.register(HeldbackPromptsInternal, f"{HELDBACK_FILE_NAME}-no_eval-temp_{temp}", temp)

"""
# TODO: DELETE
from modelgauge.config import load_secrets_from_config
from modelgauge.dependency_helper import FromSourceDependencyHelper
import os

secrets = load_secrets_from_config()
test = TESTS.make_instance("airr_official_1.0_heldback_prompt_set_internal_use_only-no_eval-temp_0.01", secrets=secrets)

data_dir = "run_data"
test_data_path = os.path.join(data_dir, "tests", test.__class__.__name__)
dependency_helper = FromSourceDependencyHelper(
        os.path.join(test_data_path, "dependency_data"),
        test.get_dependencies(),
        required_versions={},
    )
test_items = test.make_test_items(dependency_helper)

print(f"{len(test_items)}, # items")
for item in test_items[:3]:
    print(item.model_dump_json(indent=4))
    print()
"""

