from abc import ABC
from typing import Dict, List, Mapping

import pyarrow.parquet as pq  # type: ignore
from pydantic import BaseModel

from modelgauge.aggregations import get_measurement_stats
from modelgauge.annotator_set import AnnotatorSet
from modelgauge.base_test import PromptResponseTest
from modelgauge.default_annotator_set import DEFAULT_ANNOTATOR_SET
from modelgauge.dependency_helper import DependencyHelper
from modelgauge.external_data import ExternalData, WebData
from modelgauge.prompt import TextPrompt
from modelgauge.secret_values import InjectSecret, RequiredSecret, SecretDescription
from modelgauge.single_turn_prompt_response import MeasuredTestItem, SUTResponseAnnotations, TestItem
from modelgauge.sut import SUTOptions
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.test_decorator import modelgauge_test
from modelgauge.test_registry import TESTS


class RequiredModellabFileDownloadToken(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="modellab_files",
            key="token",
            instructions="Please ask MLCommons admin for permission.",
        )


class SecurityTestResult(BaseModel):
    frac_safe: float
    num_items: int


class BaseSecurityTest(PromptResponseTest, ABC):
    annotators: AnnotatorSet  # Must be set by concrete subclasses.

    def __init__(
        self,
        uid: str,
        token: RequiredModellabFileDownloadToken,
    ):
        super().__init__(uid)
        self.token = token.value

    def sut_options(self) -> SUTOptions:
        return SUTOptions(
            max_tokens=500,
            temperature=0.01,
        )

    @classmethod
    def get_annotators(cls) -> List[str]:
        return cls.annotators.annotators

    def get_dependencies(self) -> Mapping[str, ExternalData]:
        # TODO: Move prompts to prod when we get real ones. This is just a small sample file..
        prompt_set_base_download_url = "https://modellab.dev.modelmodel.org/files/download"
        headers = None
        if self.token is not None:
            headers = {"auth-token": self.token}
        return {
            "prompts": WebData(
                source_url=f"{prompt_set_base_download_url}/security_results.parquet",
                headers=headers,
            )
        }

    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        test_items: List[TestItem] = []
        local_file = dependency_helper.get_local_path("prompts")
        parquet_file = pq.ParquetFile(local_file)
        for batch in parquet_file.iter_batches(batch_size=512):
            for row in batch.to_pylist():
                test_items.append(
                    TestItem(
                        prompt=TextPrompt(text=row["attack_prompt_text"]),
                        source_id=row["release_prompt_id"],
                    ),
                )
        return test_items

    def measure_quality(self, item: SUTResponseAnnotations) -> Dict[str, float]:
        return self.annotators.evaluate(item)

    def aggregate_measurements(self, items: List[MeasuredTestItem]) -> SecurityTestResult:
        result = get_measurement_stats("is_safe", items)
        return SecurityTestResult(frac_safe=result.mean, num_items=result.count)


@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
class SecurityTest(BaseSecurityTest):
    annotators = DEFAULT_ANNOTATOR_SET


TESTS.register(SecurityTest, "sample-security-test-default", InjectSecret(RequiredModellabFileDownloadToken))
# Try to register test with private "ensemble" annotator.
try:
    from modelgauge.private_ensemble_annotator_set import PRIVATE_ANNOTATOR_SET

    @modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
    class PrivateSafeTestVersion1(BaseSecurityTest):
        annotators = PRIVATE_ANNOTATOR_SET

    TESTS.register(SecurityTest, "sample-security-test-ensemble", InjectSecret(RequiredModellabFileDownloadToken))


except Exception as e:
    pass
