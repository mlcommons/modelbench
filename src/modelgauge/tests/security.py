import csv
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Mapping, Optional

from airrlogger.log_config import get_logger
from pydantic import BaseModel

from modelgauge.aggregations import get_measurement_stats
from modelgauge.annotation import InvalidAnnotationError, ViolationProbabilityAnnotation
from modelgauge.annotators.cheval.ids import SECURITY_ANNOTATOR_V1_0_2_UID, SECURITY_ANNOTATOR_V1_1_UID
from modelgauge.base_test import PromptResponseTest
from modelgauge.dependency_helper import DependencyHelper
from modelgauge.data_packing import ZstdDecompressor
from modelgauge.external_data import ExternalData, WebData
from modelgauge.locales import validate_locale
from modelgauge.prompt import TextPrompt
from modelgauge.prompt_sets import (
    PROMPT_SET_DOWNLOAD_URL,
    SECURITY_1_0_1_JAILBREAK_PROMPT_SETS,
    SECURITY_1_0_1_NAIVE_PROMPT_SETS,
    SECURITY_1_1_JAILBREAK_PROMPT_SETS,
    ModellabFileDownloadToken,
    prompt_set_file_base_name,
    validate_token_requirement,
    validate_prompt_set,
)
from modelgauge.secret_values import InjectSecret
from modelgauge.single_turn_prompt_response import (
    MeasuredTestItem,
    SUTResponseAnnotations,
    TestItem,
    convert_annotation_to_measurement,
)
from modelgauge.model_options import ModelOptions
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.tests.safe_v1 import Hazards
from modelgauge.test_decorator import modelgauge_test
from modelgauge.test_registry import TESTS

logger = get_logger(__name__)


class SecurityTestResult(BaseModel):
    frac_safe: float
    num_items: int


class SecurityPairedTestResult(BaseModel):
    """Summary of a Security 1.1 run with the probability evaluator: only a count.

    Each response keeps its violation probability as its measurement. The naive
    and jailbreak nonviolation scores come from those probabilities in modelbench's
    external_scoring, not from this summary.
    """

    num_items: int


class BaseSecurityTest(PromptResponseTest, ABC):
    VERSION = "1.X"
    hazards = Hazards()
    prompt_sets: dict
    persona: str
    _sut_options = ModelOptions(
        max_tokens=3000,
        max_total_output_tokens=10000,  # For reasoning SUTs.
        temperature=0.01,
    )

    @classmethod
    def create_uid(cls, locale: str, prompt_set: str, evaluator=None):
        validate_prompt_set(cls.prompt_sets, prompt_set, locale)
        validate_locale(locale)
        if evaluator is None or evaluator == "default":
            postfix = ""
        else:
            postfix = "-" + evaluator
        uid = f"security-{cls.persona}-{locale}-{prompt_set}-{cls.VERSION}{postfix}".lower()
        return uid

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

    @abstractmethod
    def test_item_from_row(self, row) -> TestItem:
        pass

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
        test_items: List[TestItem] = []
        local_file = dependency_helper.get_local_path(self.prompt_set_file_base_name)
        with open(local_file, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                locale = row["locale"].lower()
                if locale != self.locale:
                    continue
                test_items.append(self.test_item_from_row(row))
        return test_items

    def measure_quality(self, item: SUTResponseAnnotations) -> Dict[str, float]:
        annotator_uid = self.get_annotators()[0]  # These tests always only have 1 annotator.
        return convert_annotation_to_measurement(item, annotator_uid)

    def aggregate_measurements(self, items: List[MeasuredTestItem]) -> SecurityTestResult:
        result = get_measurement_stats("is_safe", items)
        return SecurityTestResult(frac_safe=result.mean, num_items=result.count)


class BasePublicSecurityTest(BaseSecurityTest, ABC):
    @classmethod
    def get_annotators(cls) -> List[str]:
        return ["llama_guard_2"]


class BaseSecurityJailbreakTest(BaseSecurityTest, ABC):
    persona = "jailbreak"

    def test_item_from_row(self, row) -> TestItem:
        hazard = self.hazards.get_hazard_from_row(row)
        attack_type_id = {"name": row["attack_type_id"]}
        bijection_mapping = row["bijection_mapping"]
        if bijection_mapping:
            attack_type_id["mapping"] = bijection_mapping
        return TestItem(
            prompt=TextPrompt(text=row["prompt_text"]),
            evaluated_prompt=TextPrompt(text=row["seed_prompt_text"]),
            source_id=row["prompt_uid"],
            context={
                "hazard": hazard,
                "attack_type_id": attack_type_id,
            },
        )


class BaseSecurityJailbreakTestV1_0_1(BaseSecurityJailbreakTest, ABC):
    VERSION = "1.0.1"
    prompt_sets = SECURITY_1_0_1_JAILBREAK_PROMPT_SETS


@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
class SecurityJailbreakTestV_1_0_1(BaseSecurityJailbreakTestV1_0_1, BasePublicSecurityTest):
    pass


class BaseSecurityJailbreakTestV1_1(BaseSecurityJailbreakTest, ABC):
    """Security 1.1 reads a paired prompt file: one naive row and one jailbreak row per source prompt.

    Columns: locale, prompt_uid, naive_prompt_uid, prompt_condition, prompt, hazard,
    attack_type_id, bijection_mapping. The evaluator sees the naive prompt for both
    rows of a pair. With the probability evaluator each response keeps its violation
    probability; with a yes/no evaluator the inherited measurement applies.
    """

    VERSION = "1.1"
    prompt_sets = SECURITY_1_1_JAILBREAK_PROMPT_SETS
    _sut_options = ModelOptions(
        max_tokens=10000,
        temperature=0.01,
    )

    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        with open(dependency_helper.get_local_path(self.prompt_set_file_base_name), "r") as csvfile:
            rows = [row for row in csv.DictReader(csvfile) if row["locale"].lower() == self.locale]
        naive_prompts = {row["naive_prompt_uid"]: row["prompt"] for row in rows if row["prompt_condition"] == "naive"}
        return [self.paired_test_item(row, naive_prompts[row["naive_prompt_uid"]]) for row in rows]

    def paired_test_item(self, row, naive_prompt: str) -> TestItem:
        attack_type_id = None
        if row["prompt_condition"] == "jailbreak":
            attack_type_id = {"name": row["attack_type_id"]}
            if row.get("bijection_mapping"):
                attack_type_id["mapping"] = row["bijection_mapping"]
        return TestItem(
            prompt=TextPrompt(text=row["prompt"]),
            evaluated_prompt=TextPrompt(text=naive_prompt),
            source_id=row["prompt_uid"],
            context={
                "hazard": self.hazards.get_hazard_from_row(row),
                "attack_type_id": attack_type_id,
                "prompt_condition": row["prompt_condition"],
                "naive_prompt_uid": row["naive_prompt_uid"],
            },
        )

    # TODO: remove the deprecated thresholding and scoring here, and the downstream
    # expectation that a per-item measurement and a per-test summary exist. Scoring
    # is always product- and version-specific science in modelbench's external_scoring
    # (a name that deserves to be more descriptive), even when it is a simple average.
    def measure_quality(self, item: SUTResponseAnnotations) -> Dict[str, float]:
        annotation = item.annotations.get(self.get_annotators()[0])
        if not isinstance(annotation, ViolationProbabilityAnnotation):
            return super().measure_quality(item)
        if not annotation.is_valid or annotation.violation_probability is None:
            raise InvalidAnnotationError("Annotation is not valid:", annotation)
        return {"violation_probability": annotation.violation_probability}

    def aggregate_measurements(self, items: List[MeasuredTestItem]):
        # Returns n. Scores come from the runner's finished items and are computed in
        # external_scoring. Nothing downstream takes input from this.
        if items and all("violation_probability" in item.measurements for item in items):
            return SecurityPairedTestResult(num_items=len(items))
        return super().aggregate_measurements(items)


@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
class SecurityJailbreakTestV_1_1(BaseSecurityJailbreakTestV1_1, BasePublicSecurityTest):
    pass


class BaseSecurityNaiveTest(BaseSecurityTest, ABC):
    persona = "naive"

    def test_item_from_row(self, row) -> TestItem:
        hazard = self.hazards.get_hazard_from_row(row)
        return TestItem(
            prompt=TextPrompt(text=row["prompt_text"]),
            source_id=row["release_prompt_id"],
            context={"hazard": hazard},
        )


class BaseSecurityNaiveTestV1_0_1(BaseSecurityNaiveTest, ABC):
    VERSION = "1.0.1"
    prompt_sets = SECURITY_1_0_1_NAIVE_PROMPT_SETS


@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
class SecurityNaiveTestV1_0_1(BaseSecurityNaiveTestV1_0_1, BasePublicSecurityTest):
    pass


def register_tests(cls, evaluator=None):
    for prompt_set, prompt_set_files in cls.prompt_sets.items():
        for locale in prompt_set_files.keys():
            test_uid = cls.create_uid(locale, prompt_set, evaluator)
            TESTS.register(cls, test_uid, locale, prompt_set, InjectSecret(ModellabFileDownloadToken))


register_tests(SecurityJailbreakTestV_1_0_1)
register_tests(SecurityJailbreakTestV_1_1)
register_tests(SecurityNaiveTestV1_0_1)


# Register test with private annotator.
@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
class PrivateSecurityJailbreakTestV1_0_1(BaseSecurityJailbreakTestV1_0_1):
    @classmethod
    def get_annotators(cls) -> List[str]:
        return [SECURITY_ANNOTATOR_V1_0_2_UID]


@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
class PrivateSecurityJailbreakTestV1_1(BaseSecurityJailbreakTestV1_1):
    @classmethod
    def get_annotators(cls) -> List[str]:
        return [SECURITY_ANNOTATOR_V1_1_UID]


@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
class PrivateSecurityNaiveTestV1_0_1(BaseSecurityNaiveTestV1_0_1):
    @classmethod
    def get_annotators(cls) -> List[str]:
        return [SECURITY_ANNOTATOR_V1_0_2_UID]


register_tests(PrivateSecurityJailbreakTestV1_0_1, "private")
register_tests(PrivateSecurityJailbreakTestV1_1, "private")
register_tests(PrivateSecurityNaiveTestV1_0_1, "private")
