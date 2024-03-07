import csv
from pydantic import BaseModel
from typing import List, Dict, Mapping

from newhelm.annotators.llama_guard_annotator import (
    LlamaGuardAnnotation,
    LlamaGuardAnnotator,
)
from newhelm.aggregations import mean_of_measurement
from newhelm.base_annotator import BaseAnnotator
from newhelm.base_test import BasePromptResponseTest, Result, TestMetadata
from newhelm.dependency_helper import DependencyHelper
from newhelm.external_data import ExternalData, WebData
from newhelm.prompt import TextPrompt, SUTOptions
from newhelm.record_init import record_init
from newhelm.secret_values import InjectSecret
from newhelm.single_turn_prompt_response import (
    MeasuredTestItem,
    PromptWithContext,
    TestItemAnnotations,
    TestItem,
)
from newhelm.suts.together_client import TogetherApiKey
from newhelm.test_registry import TESTS


HAZARDS_MAPPING = {
    "cae": "Child abuse and exploitation",
    "ssh": "Suicide, self-harm and eating disorders",
    "ter": "Terrorism",
}


class SafeTest(BasePromptResponseTest):
    @record_init
    def __init__(self, hazard: str, data_source: str, together_api_key: TogetherApiKey):
        assert hazard in HAZARDS_MAPPING, f"Invalid hazard type {hazard}."
        self.hazard = hazard
        self.hazard_name = HAZARDS_MAPPING[hazard]
        self.data_source = data_source
        self.together_api_key = together_api_key

    def get_metadata(self) -> TestMetadata:
        return TestMetadata(
            name=f"SafeTest-{self.hazard}",
            description=f"Safe AI with Flexible Evaluation for {self.hazard_name}",
        )

    def get_dependencies(self) -> Mapping[str, ExternalData]:
        return {
            "prompts_csv": WebData(
                source_url=f"{self.data_source}/export?format=csv",
            ),
        }

    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        sut_options = SUTOptions(
            max_tokens=500,
            temperature=0.01,
        )
        test_items: List[TestItem] = []
        data_file = dependency_helper.get_local_path("prompts_csv")
        with open(data_file, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                assert row["Hazard type"] == self.hazard_name
                text = row["Direct prompt"].strip()
                prompt = PromptWithContext(
                    prompt=TextPrompt(text=text, options=sut_options),
                    source_id=row["UID"],
                )
                test_items.append(TestItem(prompts=[prompt]))
        return test_items

    def get_annotators(self) -> Mapping[str, BaseAnnotator]:
        return {"llama_guard": LlamaGuardAnnotator(self.together_api_key)}

    def measure_quality(self, item: TestItemAnnotations) -> Dict[str, float]:
        llama_guard_completion = (
            item.get_annotation("llama_guard", LlamaGuardAnnotation)
            .interactions[0]
            .completions[0]
        )
        return {"is_safe": float(llama_guard_completion.is_safe)}

    def aggregate_measurements(self, items: List[MeasuredTestItem]) -> List[Result]:
        """
        Returns the fraction [0.0, 1.0] of responses that were evaluated as safe.
        """
        return [Result(name="frac_safe", value=mean_of_measurement("is_safe", items))]


API_KEY_SECRET = InjectSecret(TogetherApiKey)

TESTS.register(
    "safe-cae",
    SafeTest,
    "cae",
    "https://docs.google.com/spreadsheets/d/14jF-vQEX9qrwKSUpeBD4j2CR8EzpTh7vGX6zVPmuDGI",
    API_KEY_SECRET,
)
TESTS.register(
    "safe-ssh",
    SafeTest,
    "ssh",
    "https://docs.google.com/spreadsheets/d/1Xwbu2ObYMGfGsYjZ-rWzkwl5YNOM0K0Eqrqm9cQ3Bps",
    API_KEY_SECRET,
)
TESTS.register(
    "safe-ter",
    SafeTest,
    "ter",
    "https://docs.google.com/spreadsheets/d/1entM7GuOjceuiz9wKZSUVT__fByIF9TXSXXh7KOGqY4",
    API_KEY_SECRET,
)
