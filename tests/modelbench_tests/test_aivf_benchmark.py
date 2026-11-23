import pytest

from modelbench.benchmarks_aivf import AiVerifyBenchmark, AiVerifyHazard
from modelgauge.locales import EN_US
from modelgauge_tests.fake_dependency_helper import make_csv, FakeDependencyHelper


@pytest.mark.parametrize("prompt_set", ["official"])
def test_benchmark_definition_basics(prompt_set, fake_secrets):
    b = AiVerifyBenchmark("en_us", prompt_set, "official")
    assert b.uid == f"ai_verify_benchmark-1.3-en_us-{prompt_set}-official"
    assert b.name() == "AI Verify Benchmark"
    assert b.path_name() == f"ai_verify_benchmark-1_3-en_us-{prompt_set}-official"
    h = b.hazards()
    all_hazard_keys = ["vcr", "ncr", "cse", "spc", "hte", "ssh", "sxc"]
    assert len(h) == len(all_hazard_keys)
    for hazard, hazard_key in zip(h, all_hazard_keys):
        assert isinstance(hazard, AiVerifyHazard)
        assert hazard.hazard_key == hazard_key
        assert hazard.locale == EN_US
        assert hazard.prompt_set == prompt_set


def test_annotators():
    b = AiVerifyBenchmark("en_us", "official", "private")
    for h in b.hazards():
        assert h.uid == f"ai_verify_hazard-1.3-{h.hazard_key}-en_us-official"
        for t in h.tests({}):
            assert t.get_annotators() == ["safety-v1.2.1"]
            assert t.initialization_record


def test_prompt_loading(tmp_path):
    csv_header = ["prompt_uid", "prompt_text", "hazard", "locale", "persona"]
    rows = [
        ["1", "a", "vcr", "en_us", "normal"],
        ["2", "b", "ncr", "en_us", "normal"],
        ["3", "c", "cse", "en_us", "normal"],
        ["4", "d", "spc", "en_us", "normal"],
        ["5", "e", "hte", "en_us", "normal"],
        ["6", "f", "ssh", "en_us", "normal"],
        ["7", "g", "sxc", "en_us", "normal"],
    ]
    dependencies = {"ai_verify_benchmark-1.3-en_us-official_prompt_set": make_csv(csv_header, rows)}
    dependency_helper = FakeDependencyHelper(tmp_path, dependencies)

    b = AiVerifyBenchmark("en_us", "official", "private")
    for h in b.hazards():
        assert h.uid == f"ai_verify_hazard-1.3-{h.hazard_key}-en_us-official"
        for t in h.tests({}):
            helper = dependency_helper
            t.make_test_items(helper)
            assert t.get_annotators() == ["safety-v1.2.1"]
            assert t.initialization_record
