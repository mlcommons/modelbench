import pytest

from modelgauge.annotation import MissingAnnotationError
from modelgauge.annotators.llama_guard_annotator import LlamaGuardAnnotation
from modelgauge.dependency_helper import DependencyHelper
from modelgauge.locales import EN_US
from modelgauge.prompt import TextPrompt
from modelgauge.prompt_sets import ModellabFileDownloadToken
from modelgauge.single_turn_prompt_response import MeasuredTestItem, SUTResponseAnnotations, TestItem
from modelgauge.sut import SUTResponse
from modelgauge.tests.security import SecurityJailbreakTest, SecurityNaiveTest


def _make_prompts_file(content, tmp_path):
    tmp_dir = tmp_path.mktemp("data")
    file_path = tmp_dir / "prompts.csv"
    file_path.write_text(content)
    return file_path


@pytest.fixture(scope="module")
def prompts_file_jailbreak(tmp_path_factory):
    content = (
        f"prompt_uid,prompt_text,hazard,locale,seed_prompt_text\n"
        "001,prompt 1,cse,en_US,seed 1\n"
        "002,prompt 2,cse,en_US,seed 2\n"
    )
    return _make_prompts_file(content, tmp_path_factory)


@pytest.fixture(scope="module")
def prompts_file_naive(tmp_path_factory):
    content = f"release_prompt_id,prompt_text,hazard,locale\n" "001,prompt 1,cse,en_US\n" "002,prompt 2,cse,en_US\n"
    return _make_prompts_file(content, tmp_path_factory)


class FakeDependencyHelper(DependencyHelper):
    def __init__(self, prompts_file):
        self.prompts_file = prompts_file

    def get_local_path(self, dependency_key: str):
        return self.prompts_file

    def versions_used(self):
        pass

    def update_all_dependencies(self):
        pass


@pytest.fixture
def dependency_helper_jailbreak(prompts_file_jailbreak):
    return FakeDependencyHelper(prompts_file_jailbreak)


@pytest.fixture
def dependency_helper_naive(prompts_file_naive):
    return FakeDependencyHelper(prompts_file_naive)


@pytest.fixture
def security_jailbreak_test():
    return SecurityJailbreakTest("fake-test", EN_US, "official", ModellabFileDownloadToken("fake-token"))


@pytest.fixture
def security_naive_test():
    return SecurityNaiveTest("fake-test", EN_US, "official", ModellabFileDownloadToken("fake-token"))


def test_create_uid_jailbreak():
    uid = SecurityJailbreakTest.create_uid(EN_US, "official")
    assert uid == "security-jailbreak-en_us-official-0.5"

    private_uid = SecurityJailbreakTest.create_uid(EN_US, "official", "private")
    assert private_uid == "security-jailbreak-en_us-official-0.5-private"


def test_create_uid_naive():
    uid = SecurityNaiveTest.create_uid(EN_US, "official")
    assert uid == "security-naive-en_us-official-1.0"

    private_uid = SecurityNaiveTest.create_uid(EN_US, "official", "private")
    assert private_uid == "security-naive-en_us-official-1.0-private"


def test_make_test_items_jailbreak(dependency_helper_jailbreak, security_jailbreak_test):
    items = security_jailbreak_test.make_test_items(dependency_helper_jailbreak)
    items.sort(key=lambda x: x.source_id)

    assert len(items) == 2
    assert items[0].source_id == "001"
    assert items[0].prompt.text == "prompt 1"
    assert items[0].evaluated_prompt.text == "seed 1"
    assert items[0].context["hazard"] == "cse"
    assert items[1].source_id == "002"
    assert items[1].prompt.text == "prompt 2"
    assert items[1].evaluated_prompt.text == "seed 2"
    assert items[1].context["hazard"] == "cse"


def test_make_test_items_naive(dependency_helper_naive, security_naive_test):
    items = security_naive_test.make_test_items(dependency_helper_naive)
    items.sort(key=lambda x: x.source_id)

    assert len(items) == 2
    assert all(isinstance(item, TestItem) for item in items)
    assert items[0].source_id == "001"
    assert items[0].prompt.text == "prompt 1"
    assert items[0].evaluated_prompt.text == "prompt 1"
    assert items[0].context["hazard"] == "cse"
    assert items[1].source_id == "002"
    assert items[1].prompt.text == "prompt 2"
    assert items[1].evaluated_prompt.text == "prompt 2"
    assert items[1].context["hazard"] == "cse"


def test_default_get_annotators(security_naive_test, security_jailbreak_test):
    assert security_naive_test.get_annotators() == ["llama_guard_2"]
    assert security_jailbreak_test.get_annotators() == ["llama_guard_2"]


def _test_measure_quality(security_test):
    for is_safe in [0.0, 1.0]:
        annotation = LlamaGuardAnnotation(is_safe=bool(is_safe), is_safe_logprob=0.0, violation_categories=[])
        sut_response_annotations = SUTResponseAnnotations(
            test_item=TestItem(prompt=TextPrompt(text=""), source_id=""),
            sut_response=SUTResponse(text=""),
            annotations={"llama_guard_2": annotation},
        )

        assert security_test.measure_quality(sut_response_annotations) == {"is_safe": float(is_safe)}
    # Check empty annotations
    with pytest.raises(MissingAnnotationError, match="Missing annotation for llama_guard_2"):
        security_test.measure_quality(
            SUTResponseAnnotations(
                test_item=TestItem(prompt=TextPrompt(text=""), source_id=""),
                sut_response=SUTResponse(text=""),
                annotations={},
            )
        )


def test_measure_quality(security_jailbreak_test, security_naive_test):
    _test_measure_quality(security_jailbreak_test)
    _test_measure_quality(security_naive_test)


def make_measured_item(is_safe: float):
    return MeasuredTestItem(
        test_item=TestItem(prompt=TextPrompt(text=""), source_id=""), measurements={"is_safe": is_safe}
    )


@pytest.mark.parametrize(
    "security_test",
    [
        SecurityJailbreakTest("fake-test", EN_US, "official", ModellabFileDownloadToken("fake-token")),
        SecurityNaiveTest("fake-test", EN_US, "official", ModellabFileDownloadToken("fake-token")),
    ],
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
