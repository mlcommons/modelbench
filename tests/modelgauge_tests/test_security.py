import pandas as pd
import pytest

from modelgauge.annotation import Annotation
from modelgauge.annotators.llama_guard_annotator import LlamaGuardAnnotation
from modelgauge.dependency_helper import DependencyHelper
from modelgauge.prompt import TextPrompt
from modelgauge.single_turn_prompt_response import MeasuredTestItem, SUTResponseAnnotations, TestItem
from modelgauge.sut import SUTResponse
from modelgauge.tests.security import SecurityTest


@pytest.fixture(scope="module")
def prompts_file(tmp_path_factory):
    df = pd.DataFrame(
        {
            "release_prompt_id": ["001", "002"],
            "attack_prompt_text": ["prompt 1", "prompt 2"],
        }
    )
    tmp_dir = tmp_path_factory.mktemp("data")
    file_path = tmp_dir / "prompts.parquet"
    df.to_parquet(file_path)
    return file_path


@pytest.fixture
def dependency_helper(prompts_file):
    class FakeDependencyHelper(DependencyHelper):
        def get_local_path(self, dependency_key: str):
            return prompts_file

        def versions_used(self):
            pass

        def update_all_dependencies(self):
            pass

    return FakeDependencyHelper()


def test_make_test_items(dependency_helper):
    test = SecurityTest("fake-test")
    items = test.make_test_items(dependency_helper)
    items.sort(key=lambda x: x.source_id)

    assert len(items) == 2
    assert items[0].source_id == "001"
    assert items[0].prompt.text == "prompt 1"
    assert items[1].source_id == "002"
    assert items[1].prompt.text == "prompt 2"


@pytest.mark.parametrize("is_safe", [0.0, 1.0])
def test_measure_quality(is_safe):
    test = SecurityTest("fake-test")
    annotation = Annotation.from_instance(
        LlamaGuardAnnotation(is_safe=bool(is_safe), is_safe_logprob=0.0, violation_categories=[])
    )
    sut_response_annotations = SUTResponseAnnotations(
        test_item=TestItem(prompt=TextPrompt(text=""), source_id=""),
        sut_response=SUTResponse(text=""),
        annotations={"llama_guard_2": annotation},
    )

    assert test.measure_quality(sut_response_annotations) == {"is_safe": float(is_safe)}


def make_measured_item(is_safe: float):
    return MeasuredTestItem(
        test_item=TestItem(prompt=TextPrompt(text=""), source_id=""), measurements={"is_safe": is_safe}
    )


def test_aggregate_measurements():
    test = SecurityTest("fake-test")
    safe_item = make_measured_item(1.0)
    unsafe_item = make_measured_item(0.0)

    # All safe.
    result = test.aggregate_measurements([safe_item, safe_item, safe_item])
    assert result.num_items == 3
    assert result.frac_safe == 1.0
    # All unsafe.
    result = test.aggregate_measurements([unsafe_item, unsafe_item, unsafe_item])
    assert result.num_items == 3
    assert result.frac_safe == 0.0
    # Mixed.
    result = test.aggregate_measurements([unsafe_item, safe_item, unsafe_item])
    assert result.num_items == 3
    assert result.frac_safe == float(1 / 3)
