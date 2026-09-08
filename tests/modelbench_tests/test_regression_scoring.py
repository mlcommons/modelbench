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


@pytest.fixture
def a_benchmark():
    return GeneralPurposeAiChatBenchmarkV1_2("en_us", "official", "private")


def test_basics(sut, a_benchmark):
    result: BenchmarkRegressionScore = RegressionScorer().score(a_benchmark, sut, fake_test_run_items(sut))

    assert result.sut is sut
    assert result.benchmark_definition is a_benchmark
    assert time.time() - result.end_time.timestamp() < 1.0

    assert result.score == pytest.approx(92.9988)  # not verified; just checks stability

    for hs in result.hazard_scores:
        assert hs.exceptions == 0
        assert hs.num_safe_items == 1
        assert hs.num_scored_items == 1
    pprint.pprint(vars(result))


def test_exception_count(sut, a_benchmark):
    items_with_exceptions = fake_test_run_items(sut, exceptions_per_hazard=1)
    result = RegressionScorer().score(a_benchmark, sut, items_with_exceptions)
    for hs in result.hazard_scores:
        assert hs.exceptions == 1


def test_no_data(sut, a_benchmark):
    with pytest.raises(ValueError):
        RegressionScorer().score(a_benchmark, sut, [])


def test_not_all_hazards(sut, a_benchmark):
    items = fake_test_run_items(sut)[0:5]
    with pytest.raises(ValueError):
        RegressionScorer().score(a_benchmark, sut, items)


def test_missing_context_data(sut, a_benchmark):
    items = fake_test_run_items(sut)
    for item in items:
        item.test_item.context_internal = SafeTestItemContext(
            persona_type="fnord", hazard=item.test_item.context.hazard
        )
    with pytest.raises(AssertionError):
        RegressionScorer().score(a_benchmark, sut, items)


def fake_test_run_items(sut: FakeSUT, exceptions_per_hazard=0) -> list[TestRunItem]:
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
            [ValueError(f"fake exception {i}") for i in range(exceptions_per_hazard)],
        )
        test_run_items.append(tri)
    return test_run_items
