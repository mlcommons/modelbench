import pprint
import random
import time
from unittest.mock import MagicMock

import pytest

from modelbench.benchmark_runner_items import TestRunItem
from modelbench.benchmark_score import BenchmarkRegressionScore
from modelbench.benchmarks import GeneralPurposeAiChatBenchmarkV1_2
from modelbench.external_scoring.scoring_adapters import RegressionScorer
from modelgauge.annotation import SafetyAnnotation
from modelgauge.prompt import TextPrompt
from modelgauge.single_turn_prompt_response import TestItem
from modelgauge.tests.safe_v1 import SafeTestItemContext
from modelgauge_tests.fake_sut import FakeSUT

REQUIRED_HAZARDS = {"cse", "dfm", "hte", "ipv", "iwp", "ncr", "prv", "spc_lgl", "src", "ssh", "sxc_prn", "vcr"}

"""
    Regression scoring is what we're calling the scoring approach added in September 2026 by
    Riki Conrey, and used in Safety 1.2 and Security 1.1.
"""


def test_basics(sut):
    benchmark = GeneralPurposeAiChatBenchmarkV1_2("en_us", "official", "private")

    rs = RegressionScorer()
    result: BenchmarkRegressionScore = rs.score(benchmark, sut, fake_test_run_items(sut))

    assert result.sut is sut
    assert result.benchmark_definition is benchmark
    assert time.time() - result.end_time.timestamp() < 1.0

    assert result.score == pytest.approx(92.9988)  # not verified; just checks stability

    for hs in result.hazard_scores:
        assert hs.exceptions == 0
        assert hs.num_safe_items == 1
        assert hs.num_scored_items == 1
    pprint.pprint(vars(result))


def fake_test_run_items(sut: FakeSUT) -> list[TestRunItem]:
    test_run_items = []
    rand = random.Random(0)

    for n, hazard in enumerate(REQUIRED_HAZARDS):
        prompt_id = f"prompt-{n}"
        tri = TestRunItem(
            MagicMock(),
            TestItem(
                prompt=TextPrompt(text="6*7="),
                source_id=prompt_id,
                context=SafeTestItemContext(
                    persona_type="fnord",
                    hazard=hazard,
                    difficulty_b=rand.uniform(-3.22638750076, 3.358253479),
                    discrimination_a=rand.uniform(1.58203667959, 2.04493841643),
                ),
            ),
            sut,
            MagicMock(),
            {"thingy": SafetyAnnotation(is_safe=True, is_valid=True)},
            MagicMock(),
            MagicMock(),
        )
        test_run_items.append(tri)
    return test_run_items


# TODO test edge cases: no data, wrong context, ?
