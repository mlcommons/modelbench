from typing import Dict, List, Mapping

import pyarrow.parquet as pq  # type: ignore
from pydantic import BaseModel

from modelgauge.aggregations import get_measurement_stats
from modelgauge.base_test import PromptResponseTest
from modelgauge.default_annotator_set import DEFAULT_ANNOTATOR_SET
from modelgauge.dependency_helper import DependencyHelper
from modelgauge.external_data import ExternalData, LocalData
from modelgauge.prompt import TextPrompt
from modelgauge.single_turn_prompt_response import MeasuredTestItem, SUTResponseAnnotations, TestItem
from modelgauge.sut import SUTOptions
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.test_decorator import modelgauge_test
from modelgauge.test_registry import TESTS


class SecurityTestResult(BaseModel):
    frac_safe: float
    num_items: int


@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
class SecurityTest(PromptResponseTest):
    # TODO: Replace with ensemble?
    annotators = DEFAULT_ANNOTATOR_SET

    def sut_options(self) -> SUTOptions:
        # TODO: Verify
        return SUTOptions(
            max_tokens=500,
            temperature=0.01,
        )

    @classmethod
    def get_annotators(cls) -> List[str]:
        return cls.annotators.annotators

    def get_dependencies(self) -> Mapping[str, ExternalData]:
        # TODO: Move prompts to web.
        return {"prompts": LocalData(path="results.parquet")}

    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        test_items: List[TestItem] = []
        local_file = dependency_helper.get_local_path("prompts")
        parquet_file = pq.ParquetFile(local_file)
        # TODO: Up batch size.
        for batch in parquet_file.iter_batches(batch_size=1):
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
        """Combine the measurements for each TestItem into a test specific Typeable."""
        result = get_measurement_stats("is_safe", items)
        return SecurityTestResult(frac_safe=result.mean, num_items=result.count)


TESTS.register(SecurityTest, "demo-security-test")
