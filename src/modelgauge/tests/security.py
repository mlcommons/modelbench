import csv
from abc import ABC
from typing import Dict, List, Mapping, Optional

from pydantic import BaseModel

from modelgauge.aggregations import get_measurement_stats
from modelgauge.annotator_set import AnnotatorSet
from modelgauge.base_test import PromptResponseTest
from modelgauge.default_annotator_set import DEFAULT_ANNOTATOR_SET
from modelgauge.dependency_helper import DependencyHelper
from modelgauge.external_data import ExternalData, WebData
from modelgauge.locales import validate_locale
from modelgauge.prompt import TextPrompt
from modelgauge.prompt_sets import (
    GENERAL_PROMPT_SETS,
    PROMPT_SET_DOWNLOAD_URL,
    SECURITY_JAILBREAK_PROMPT_SETS,
    ModellabFileDownloadToken,
    prompt_set_file_base_name,
    validate_token_requirement,
    validate_prompt_set,
)
from modelgauge.secret_values import InjectSecret
from modelgauge.single_turn_prompt_response import MeasuredTestItem, SecurityContext, SUTResponseAnnotations, TestItem
from modelgauge.sut import SUTOptions
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.tests.safe_v1 import Hazards
from modelgauge.test_decorator import modelgauge_test
from modelgauge.test_registry import TESTS


class SecurityTestResult(BaseModel):
    frac_safe: float
    num_items: int


class BaseSecurityTest(PromptResponseTest, ABC):
    annotators: AnnotatorSet  # Must be set by concrete subclasses.
    hazards = Hazards()
    prompt_sets: dict
    persona: str

    @classmethod
    def create_uid(cls, locale: str, prompt_set: str, evaluator=None):
        validate_prompt_set(cls.prompt_sets, prompt_set, locale)
        validate_locale(locale)
        if evaluator is None or evaluator == "default":
            postfix = ""
        else:
            postfix = "-" + evaluator
        uid = f"security-{cls.persona}-{locale}-{prompt_set}-0.5{postfix}".lower()
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

    def sut_options(self) -> SUTOptions:
        return SUTOptions(
            max_tokens=3000,
            max_total_output_tokens=10000,  # For reasoning SUTs.
            temperature=0.01,
        )

    @classmethod
    def get_annotators(cls) -> List[str]:
        return cls.annotators.annotators

    def get_dependencies(self) -> Mapping[str, ExternalData]:
        headers = None
        if self.token is not None:
            headers = {"auth-token": self.token.value}
        source_url = f"{PROMPT_SET_DOWNLOAD_URL}/{self.prompt_set_file_base_name}.csv"
        return {
            self.prompt_set_file_base_name: WebData(
                source_url=source_url,
                headers=headers,
            )
        }

    def measure_quality(self, item: SUTResponseAnnotations) -> Dict[str, float]:
        return self.annotators.evaluate(item)

    def aggregate_measurements(self, items: List[MeasuredTestItem]) -> SecurityTestResult:
        result = get_measurement_stats("is_safe", items)
        return SecurityTestResult(frac_safe=result.mean, num_items=result.count)


class BaseSecurityJailbreakTest(BaseSecurityTest, ABC):
    prompt_sets = SECURITY_JAILBREAK_PROMPT_SETS
    persona = "jailbreak"

    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        test_items: List[TestItem] = []
        local_file = dependency_helper.get_local_path(self.prompt_set_file_base_name)
        with open(local_file, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                hazard = self.hazards.get_hazard_from_row(row)
                locale = row["locale"].lower()
                if locale != self.locale:
                    continue
                test_items.append(
                    TestItem(
                        prompt=TextPrompt(text=row["prompt_text"]),
                        source_id=row["prompt_uid"],
                        context=SecurityContext(seed_prompt=row["seed_prompt_text"], hazard=hazard),
                    ),
                )
        return test_items


@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
class SecurityJailbreakTest(BaseSecurityJailbreakTest):
    annotators = DEFAULT_ANNOTATOR_SET


class BaseSecurityNaiveTest(BaseSecurityTest):
    prompt_sets = GENERAL_PROMPT_SETS
    persona = "naive"

    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        test_items: List[TestItem] = []
        local_file = dependency_helper.get_local_path(self.prompt_set_file_base_name)
        csvfile = open(local_file, "r")
        reader = csv.DictReader(csvfile)
        for row in reader:
            hazard = self.hazards.get_hazard_from_row(row)
            locale = row["locale"].lower()
            if locale != self.locale:
                continue
            test_items.append(
                TestItem(
                    prompt=TextPrompt(text=row["prompt_text"]),
                    source_id=row["release_prompt_id"],
                    context={"hazard": hazard},
                ),
            )
        return test_items


@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
class SecurityNaiveTest(BaseSecurityNaiveTest):
    annotators = DEFAULT_ANNOTATOR_SET


def register_tests(cls, evaluator=None):
    for prompt_set, prompt_set_files in cls.prompt_sets.items():
        for locale in prompt_set_files.keys():
            test_uid = cls.create_uid(locale, prompt_set, evaluator)
            TESTS.register(cls, test_uid, locale, prompt_set, InjectSecret(ModellabFileDownloadToken))


register_tests(SecurityJailbreakTest)
register_tests(SecurityNaiveTest)

# Try to register test with private "ensemble" annotator.
try:
    from modelgauge.private_ensemble_annotator_set import PRIVATE_SECURITY_ANNOTATOR_SET

    @modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
    class PrivateSecurityJailbreakTest(BaseSecurityJailbreakTest):
        annotators = PRIVATE_SECURITY_ANNOTATOR_SET

    @modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
    class PrivateSecurityNaiveTest(BaseSecurityNaiveTest):
        annotators = PRIVATE_SECURITY_ANNOTATOR_SET

    register_tests(PrivateSecurityJailbreakTest, "ensemble")
    register_tests(PrivateSecurityNaiveTest, "ensemble")

except Exception as e:
    pass
