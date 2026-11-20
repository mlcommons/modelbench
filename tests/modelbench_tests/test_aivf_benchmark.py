import pytest

from modelbench.benchmarks_aivf import AiVerifyBenchmark, AiVerifyHazard
from modelgauge.locales import EN_SG


@pytest.mark.parametrize("prompt_set", ["official"])
def test_benchmark_definition_basics(prompt_set, fake_secrets):
    b = AiVerifyBenchmark("en_sg", prompt_set)
    assert b.uid == f"ai_verify_benchmark-1.3-en_sg-{prompt_set}-default"
    assert b.name() == "AI Verify Benchmark"
    assert b.path_name() == f"ai_verify_benchmark-1_3-en_sg-{prompt_set}-default"
    h = b.hazards()
    all_hazard_keys = ["vcr", "ncr", "cse", "spc", "hte", "ssh", "sxc"]
    assert len(h) == len(all_hazard_keys)
    for hazard, hazard_key in zip(h, all_hazard_keys):
        assert isinstance(hazard, AiVerifyHazard)
        assert hazard.hazard_key == hazard_key
        assert hazard.locale == EN_SG
        assert hazard.prompt_set == prompt_set
