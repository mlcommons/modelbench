import pytest

from modelgauge.annotation import Annotation
from modelgauge.annotators.llama_guard_annotator import LlamaGuardAnnotation
from modelgauge.dependency_helper import DependencyHelper
from modelgauge.locales import EN_US
from modelgauge.prompt import TextPrompt
from modelgauge.prompt_sets import ModellabFileDownloadToken
from modelgauge.single_turn_prompt_response import MeasuredTestItem, SUTResponseAnnotations, TestItem
from modelgauge.sut import SUTResponse
from modelgauge.tests.security import SecurityTest

HAZARD = "cse"


@pytest.fixture(scope="module")
def prompts_file(tmp_path_factory):
    content = (
        f"prompt_uid,prompt_text,hazard,locale,seed_prompt_text\n"
        "001,prompt 1,cse,en_US,seed 1\n"
        "002,prompt 2,cse,en_US,seed 2\n"
    )
    tmp_dir = tmp_path_factory.mktemp("data")
    file_path = tmp_dir / "prompts.csv"
    file_path.write_text(content)
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


@pytest.fixture
def security_test():
    return SecurityTest("fake-test", EN_US, "official", ModellabFileDownloadToken("fake-token"))


def test_create_uid():
    uid = SecurityTest.create_uid(EN_US, "official")
    assert uid == "security-jailbreak-en_us-official-0.5"

    private_uid = SecurityTest.create_uid(EN_US, "official", "ensemble")
    assert private_uid == "security-jailbreak-en_us-official-0.5-ensemble"


def test_make_test_items(dependency_helper, security_test):
    items = security_test.make_test_items(dependency_helper)
    items.sort(key=lambda x: x.source_id)

    assert len(items) == 2
    assert items[0].source_id == "001"
    assert items[0].prompt.text == "prompt 1"
    assert items[0].context.seed_prompt == "seed 1"
    assert items[0].context.hazard == "cse"
    assert items[1].source_id == "002"
    assert items[1].prompt.text == "prompt 2"
    assert items[1].context.seed_prompt == "seed 2"
    assert items[1].context.hazard == "cse"


@pytest.mark.parametrize("is_safe", [0.0, 1.0])
def test_measure_quality(is_safe, security_test):
    annotation = Annotation.from_instance(
        LlamaGuardAnnotation(is_safe=bool(is_safe), is_safe_logprob=0.0, violation_categories=[])
    )
    sut_response_annotations = SUTResponseAnnotations(
        test_item=TestItem(prompt=TextPrompt(text=""), source_id=""),
        sut_response=SUTResponse(text=""),
        annotations={"llama_guard_2": annotation},
    )

    assert security_test.measure_quality(sut_response_annotations) == {"is_safe": float(is_safe)}


def make_measured_item(is_safe: float):
    return MeasuredTestItem(
        test_item=TestItem(prompt=TextPrompt(text=""), source_id=""), measurements={"is_safe": is_safe}
    )


def test_aggregate_measurements(security_test):
    safe_item = make_measured_item(1.0)
    unsafe_item = make_measured_item(0.0)

    # All safe.
    result = security_test.aggregate_measurements([safe_item, safe_item, safe_item])
    assert result.num_items == 3
    assert result.frac_safe == 1.0
    # All unsafe.
    result = security_test.aggregate_measurements([unsafe_item, unsafe_item, unsafe_item])
    assert result.num_items == 3
    assert result.frac_safe == 0.0
    # Mixed.
    result = security_test.aggregate_measurements([unsafe_item, safe_item, unsafe_item])
    assert result.num_items == 3
    assert result.frac_safe == float(1 / 3)
