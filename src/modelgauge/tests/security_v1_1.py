import csv
from typing import Dict, List, Mapping, Optional

from pydantic import BaseModel

from modelgauge.annotation import InvalidAnnotationError, ViolationProbabilityAnnotation
from modelgauge.annotators.cheval.ids import SECURITY_ANNOTATOR_V1_1_UID
from modelgauge.base_test import PromptResponseTest
from modelgauge.data_packing import ZstdDecompressor
from modelgauge.dependency_helper import DependencyHelper
from modelgauge.external_data import ExternalData, WebData
from modelgauge.locales import validate_locale
from modelgauge.model_options import ModelOptions
from modelgauge.prompt import TextPrompt
from modelgauge.prompt_sets import (
    PROMPT_SET_DOWNLOAD_URL,
    SECURITY_1_1_JAILBREAK_PROMPT_SETS,
    ModellabFileDownloadToken,
    prompt_set_file_base_name,
    validate_prompt_set,
    validate_token_requirement,
)
from modelgauge.single_turn_prompt_response import MeasuredTestItem, SUTResponseAnnotations, TestItem
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.test_decorator import modelgauge_test
from modelgauge.tests.safe_v1 import Hazards
from modelgauge.tests.security_v1_0 import register_tests


class NullTestResult(BaseModel):
    """An empty test result for benchmarks that aggregate measurements in their own scoring code."""


@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
class SecurityTestV1_1(PromptResponseTest):
    """Security 1.1 reads a paired prompt file: one naive row and one jailbreak row per source prompt.

    Columns: locale, prompt_uid, naive_prompt_uid, prompt_condition, prompt, hazard,
    attack_type_id, bijection_mapping. The evaluator sees the naive prompt for both
    rows of a pair.
    """

    VERSION = "1.1"
    hazards = Hazards()
    prompt_sets = SECURITY_1_1_JAILBREAK_PROMPT_SETS
    _sut_options = ModelOptions(
        max_tokens=10000,
        temperature=0.01,
    )

    @classmethod
    def create_uid(cls, locale: str, prompt_set: str, evaluator=None):
        validate_prompt_set(cls.prompt_sets, prompt_set, locale)
        validate_locale(locale)
        postfix = "" if evaluator is None or evaluator == "default" else "-" + evaluator
        return f"security-{locale}-{prompt_set}-{cls.VERSION}{postfix}".lower()

    def __init__(
        self,
        uid: str,
        locale: str,
        prompt_set: str,
        token: Optional[ModellabFileDownloadToken] = None,
    ):
        super().__init__(uid)
        validate_prompt_set(self.prompt_sets, prompt_set, locale)
        validate_token_requirement(prompt_set, token)
        validate_locale(locale)
        self.token = token
        self.locale = locale
        self.prompt_set_file_base_name = prompt_set_file_base_name(self.prompt_sets, prompt_set, locale)

    @classmethod
    def get_annotators(cls) -> List[str]:
        return [SECURITY_ANNOTATOR_V1_1_UID]

    def get_dependencies(self) -> Mapping[str, ExternalData]:
        headers = None
        if self.token is not None:
            headers = {"auth-token": self.token.value}
        source_url = f"{PROMPT_SET_DOWNLOAD_URL}/{self.prompt_set_file_base_name}.csv.zst"
        return {
            self.prompt_set_file_base_name: WebData(
                source_url=source_url,
                headers=headers,
                decompressor=ZstdDecompressor(),
            )
        }

    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        with open(dependency_helper.get_local_path(self.prompt_set_file_base_name), "r") as csvfile:
            rows = [row for row in csv.DictReader(csvfile) if row["locale"].lower() == self.locale]
        naive_prompts = {
            row["naive_prompt_uid"]: row["prompt_text"] for row in rows if row["prompt_condition"] == "naive"
        }
        return [self.test_item_from_row(row, naive_prompts[row["naive_prompt_uid"]]) for row in rows]

    def test_item_from_row(self, row, naive_prompt: str) -> TestItem:
        attack_type_id = None
        if row["prompt_condition"] == "jailbreak":
            attack_type_id = {"name": row["attack_type_id"]}
            if row.get("bijection_mapping"):
                attack_type_id["mapping"] = row["bijection_mapping"]
        return TestItem(
            prompt=TextPrompt(text=row["prompt_text"]),
            evaluated_prompt=TextPrompt(text=naive_prompt),
            source_id=row["prompt_uid"],
            context={
                "hazard": self.hazards.get_hazard_from_row(row),
                "attack_type_id": attack_type_id,
                "prompt_condition": row["prompt_condition"],
                "naive_prompt_uid": row["naive_prompt_uid"],
            },
        )

    def measure_quality(self, item: SUTResponseAnnotations) -> Dict[str, float]:
        annotation = item.annotations.get(self.get_annotators()[0])
        if (
            not isinstance(annotation, ViolationProbabilityAnnotation)
            or not annotation.is_valid
            or annotation.violation_probability is None
        ):
            raise InvalidAnnotationError("Annotation is not a valid violation probability:", annotation)
        return {"violation_probability": annotation.violation_probability}

    def aggregate_measurements(self, items: List[MeasuredTestItem]):
        # Item measurements are aggregated in external scoring code.
        return NullTestResult()


register_tests(SecurityTestV1_1, "private")
