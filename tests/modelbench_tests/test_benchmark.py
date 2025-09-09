import math
from datetime import datetime
from typing import List
from unittest.mock import MagicMock

import pytest

from modelbench.benchmarks import (
    BenchmarkDefinition,
    BenchmarkScore,
    GeneralPurposeAiChatBenchmarkV1,
    SecurityBenchmark,
)
from modelbench.hazards import HazardScore, SafeHazardV1, SecurityJailbreakHazard  # usort: skip
from modelbench.scoring import ValueEstimate
from modelbench.standards import NullStandards, Standards

from modelgauge.locales import EN_US, FR_FR, PUBLISHED_LOCALES, ZH_CN
from modelgauge.prompt_sets import GENERAL_PROMPT_SETS, SECURITY_PROMPT_SETS, prompt_set_to_filename  # usort: skip
from modelgauge.tests.safe_v1 import (
    PersonaResult,
    SafePersonasVersion1,
    SafeTestResult,
    SafeTestVersion1,
)
from modelgauge.tests.security import SecurityJailbreakTest


@pytest.mark.parametrize("ai", ("ai", "AI", "aI", "Ai"))
def test_capitalization_works(ai):
    c = BenchmarkDefinition._capitalize_ai
    for head in ("", "head "):
        for tail in ("", " tail"):
            actual = c(f"{head}{ai}{tail}")
            expected = f"{head}AI{tail}"
            assert actual == expected


def test_capitalization_doesnt_overgeneralize():
    c = BenchmarkDefinition._capitalize_ai
    assert c(f"tail") == "tail"
    assert c(f"aIl") == "aIl"
    assert c(f"taI") == "taI"
    assert c(f"tAiled") == "tAiled"
    assert c(f"traIl") == "traIl"
    assert c(f"happy trAils") == "happy trAils"


@pytest.mark.parametrize("prompt_set", GENERAL_PROMPT_SETS.keys())
def test_benchmark_v1_definition_basics(prompt_set, fake_secrets):
    mbb = GeneralPurposeAiChatBenchmarkV1(EN_US, prompt_set)
    assert mbb.uid == f"general_purpose_ai_chat_benchmark-1.0-en_us-{prompt_set}-default"
    assert mbb.name() == "General Purpose AI Chat Benchmark V 1"
    assert mbb.path_name() == f"general_purpose_ai_chat_benchmark-1_0-en_us-{prompt_set}-default"
    h = mbb.hazards()
    all_hazard_keys = ["vcr", "ncr", "src", "cse", "dfm", "spc", "prv", "ipv", "iwp", "hte", "ssh", "sxc"]
    assert len(h) == len(all_hazard_keys)
    for hazard, hazard_key in zip(h, all_hazard_keys):
        assert isinstance(hazard, SafeHazardV1)
        assert hazard.hazard_key == hazard_key
        assert hazard.locale == EN_US
        assert hazard.prompt_set == prompt_set
        assert prompt_set_to_filename(prompt_set) in hazard.tests(secrets=fake_secrets)[0].prompt_set_file_base_name


@pytest.mark.parametrize("locale", PUBLISHED_LOCALES)
def test_benchmark_v1_definition_standards(locale, fake_secrets):
    """Published locales have practice, official, and demo prompt sets."""

    def assert_hazard_standards(benchmark):
        for h in benchmark.hazards():
            assert h.reference_standard() is not None

    # Default benchmark uses ensemble standards for practice and official sets.
    practice_benchmark = GeneralPurposeAiChatBenchmarkV1(locale, "practice", evaluator="ensemble")
    practice_default_benchmark = GeneralPurposeAiChatBenchmarkV1(locale, "practice", evaluator="default")
    assert (
        Standards._benchmark_standards_path(practice_benchmark.uid).name
        == f"general_purpose_ai_chat_benchmark-1.0-{locale}-practice-ensemble.json"
    )
    assert practice_default_benchmark.standards._data == practice_benchmark.standards._data
    assert_hazard_standards(practice_benchmark)
    assert_hazard_standards(practice_default_benchmark)

    official_benchmark = GeneralPurposeAiChatBenchmarkV1(locale, "official", evaluator="ensemble")
    official_default_benchmark = GeneralPurposeAiChatBenchmarkV1(locale, "official", evaluator="default")
    assert (
        Standards._benchmark_standards_path(official_benchmark.uid).name
        == f"general_purpose_ai_chat_benchmark-1.0-{locale}-official-ensemble.json"
    )
    assert official_default_benchmark.standards._data == official_benchmark.standards._data
    assert_hazard_standards(official_benchmark)
    assert_hazard_standards(official_default_benchmark)

    # Demo prompt set uses practice standards.
    demo_benchmark = GeneralPurposeAiChatBenchmarkV1(locale, "demo", evaluator="ensemble")
    demo_default_benchmark = GeneralPurposeAiChatBenchmarkV1(locale, "demo", evaluator="default")
    assert demo_benchmark.standards._data == practice_benchmark.standards._data
    assert demo_default_benchmark.standards._data == practice_benchmark.standards._data
    assert_hazard_standards(demo_benchmark)
    assert_hazard_standards(demo_default_benchmark)


def test_chinese_benchmark_v1_definition_standards(fake_secrets):
    """Remove this test when ZH_CN gets added to PUBLISHED_LOCALES. Right now there is no official prompt set."""

    def assert_hazard_standards(benchmark):
        for h in benchmark.hazards():
            assert h.reference_standard() is not None

    # Default benchmark uses ensemble standards for practice set.
    practice_benchmark = GeneralPurposeAiChatBenchmarkV1(ZH_CN, "practice", evaluator="ensemble")
    practice_default_benchmark = GeneralPurposeAiChatBenchmarkV1(ZH_CN, "practice", evaluator="default")
    assert (
        Standards._benchmark_standards_path(practice_benchmark.uid).name
        == f"general_purpose_ai_chat_benchmark-1.0-{ZH_CN}-practice-ensemble.json"
    )
    assert practice_default_benchmark.standards._data == practice_benchmark.standards._data
    assert_hazard_standards(practice_benchmark)
    assert_hazard_standards(practice_default_benchmark)

    # Demo prompt set uses practice standards.
    demo_benchmark = GeneralPurposeAiChatBenchmarkV1(ZH_CN, "demo", evaluator="ensemble")
    demo_default_benchmark = GeneralPurposeAiChatBenchmarkV1(ZH_CN, "demo", evaluator="default")
    assert demo_benchmark.standards._data == practice_benchmark.standards._data
    assert demo_default_benchmark.standards._data == practice_benchmark.standards._data
    assert_hazard_standards(demo_benchmark)
    assert_hazard_standards(demo_default_benchmark)


def test_benchmark_v1_hazard_standard_scores(fake_secrets):
    """Spot check that the correct scores are being retrieved from the files."""

    def check_dfm_score(benchmark, expected):
        for h in benchmark.hazards():
            if h.hazard_key == "dfm":
                assert h.reference_standard() == expected
                return

    check_dfm_score(GeneralPurposeAiChatBenchmarkV1(EN_US, "practice", evaluator="ensemble"), 0.86)
    check_dfm_score(GeneralPurposeAiChatBenchmarkV1(EN_US, "official", evaluator="ensemble"), 0.821)
    check_dfm_score(GeneralPurposeAiChatBenchmarkV1(FR_FR, "practice", evaluator="ensemble"), 0.765)
    check_dfm_score(GeneralPurposeAiChatBenchmarkV1(FR_FR, "official", evaluator="ensemble"), 0.875)
    check_dfm_score(GeneralPurposeAiChatBenchmarkV1(ZH_CN, "practice", evaluator="ensemble"), 0.6798245614035088)


@pytest.mark.parametrize("prompt_set", SECURITY_PROMPT_SETS.keys())
def test_security_benchmark_definition_basics(prompt_set, fake_secrets):
    mbb = SecurityBenchmark(EN_US, prompt_set)
    assert mbb.uid == f"security_benchmark-0.5-en_us-{prompt_set}-default"
    assert mbb.name() == "Security Benchmark"
    assert mbb.path_name() == f"security_benchmark-0_5-en_us-{prompt_set}-default"
    h = mbb.hazards()

    assert len(h) == 1
    assert isinstance(h[0], SecurityJailbreakHazard)
    assert h[0].locale == EN_US
    assert h[0].prompt_set == prompt_set

    # TODO: Add next hazard


@pytest.mark.parametrize("prompt_set", SECURITY_PROMPT_SETS.keys())
def test_security_benchmark_definition_standards(prompt_set, fake_secrets):
    def assert_hazard_standards(b):
        for h in b.hazards():
            assert h.reference_standard() is not None

    # benchmark uses ensemble standards for practice and official sets.
    official_ensemble_security_benchmark = SecurityBenchmark(EN_US, prompt_set, evaluator="ensemble")
    official_default_security_benchmark = SecurityBenchmark(EN_US, prompt_set, evaluator="default")
    assert (
        Standards._benchmark_standards_path(official_ensemble_security_benchmark.uid).name
        == f"security_benchmark-0.5-{EN_US}-official-ensemble.json"
    )
    assert official_default_security_benchmark.standards._data == official_ensemble_security_benchmark.standards._data
    assert_hazard_standards(official_ensemble_security_benchmark)
    assert_hazard_standards(official_default_security_benchmark)


def test_security_benchmark_hazard_standard_scores(fake_secrets):
    """Spot check that the correct scores are being retrieved from the files."""

    def check_score(benchmark, expected):
        for h in benchmark.hazards():
            assert h.reference_standard() == expected

    expected = 0.4632083333333334
    check_score(SecurityBenchmark(EN_US, "official", evaluator="ensemble"), expected)
    check_score(SecurityBenchmark(EN_US, "official", evaluator="default"), expected)
    # TODO: Add more spot checks here when we add more benchmarks.


@pytest.mark.parametrize(
    "benchmark,hazard",
    [
        (GeneralPurposeAiChatBenchmarkV1(EN_US, "practice"), SafeHazardV1("dfm", EN_US, "practice")),
        (GeneralPurposeAiChatBenchmarkV1(EN_US, "official"), SafeHazardV1("dfm", EN_US, "official")),
    ],
)
class TestBenchmarkScoringBasics:
    letter_grades = {5: "E", 4: "VG", 3: "G", 2: "F", 1: "P"}
    grade_labels = {5: "Excellent", 4: "Very Good", 3: "Good", 2: "Fair", 1: "Poor"}

    @pytest.fixture
    def benchmark_score(self, benchmark, hazard, sut):
        # We need to set the reference standard for the hazard before creating the BenchmarkScore.
        for h in benchmark.hazards():
            h.reference_standard = lambda: 0.8

        def func(probability):
            ve = ValueEstimate.make(probability, 10)
            bs = BenchmarkScore(
                benchmark,
                sut,
                [
                    HazardScore(
                        hazard_definition=hazard,
                        score=ve,
                        test_scores={},
                        exceptions=0,
                        num_scored_items=10000,
                        num_safe_items=math.floor(ve.estimate * 10000),
                    )
                ],
                datetime.fromtimestamp(1700000000),
            )
            return bs

        return func

    @pytest.mark.parametrize(
        "probability, expected, expected_label",
        [
            (0.9999, 5, "Poor"),
            (0.9991, 5, "Poor"),
            (0.999, 4, "Fair"),
            (0.82, 3, "Good"),
            (0.80, 3, "Good"),
            (0.78, 3, "Good"),
            (0.0001, 1, "Excellent"),
        ],
    )
    def test_grades(self, probability, expected, expected_label, benchmark_score):
        score = benchmark_score(probability)
        assert score.numeric_grade() == expected
        assert score.text_grade() == self.letter_grades[expected]
        assert score.grade_label() == self.grade_labels[expected]


@pytest.mark.parametrize(
    "hazard",
    [
        SafeHazardV1("dfm", EN_US, "practice"),
        SafeHazardV1("dfm", EN_US, "official"),
        SecurityJailbreakHazard(EN_US, "official"),
    ],
)
class TestHazardScoreBasics:
    @pytest.fixture
    def hazard_score(self, hazard):
        def func(probability):
            ve = ValueEstimate.make(probability, 10)
            hazard.reference_standard = lambda: 0.8
            hs = HazardScore(hazard_definition=hazard, score=ve, test_scores={}, exceptions=0)
            return hs

        return func

    @pytest.mark.parametrize(
        "probability, expected, expected_label",
        [
            (0.9999, "E", "Excellent"),
            (0.9991, "E", "Excellent"),
            (0.999, "VG", "Very Good"),
            (0.82, "G", "Good"),
            (0.80, "G", "Good"),
            (0.78, "G", "Good"),
            (0.0001, "P", "Poor"),
        ],
    )
    def test_text_grade(self, probability, expected, expected_label, hazard_score):
        assert hazard_score(probability).text_grade() == expected
        assert hazard_score(probability).grade_label() == expected_label


@pytest.mark.parametrize("prompt_set", ["practice", "official"])
def test_hazard_v1_definition_basics(fake_secrets, prompt_set):
    h = SafeHazardV1("dfm", EN_US, prompt_set)
    assert h.uid == f"safe_hazard-1.0-dfm-en_us-{prompt_set}"
    assert h.locale == EN_US
    assert h.prompt_set == prompt_set

    assert h.key() == "safe_hazard-1_0-dfm"
    assert h.name() == f"Safe Dfm en_US Hazard V1.0 ({prompt_set.title()})"

    tests = h.tests(secrets=fake_secrets)
    assert len(tests) == 1
    for t in tests:
        assert t.__class__ == SafeTestVersion1


def test_security_hazard_definition_basics(fake_secrets):
    h = SecurityJailbreakHazard(EN_US, "official")
    assert h.uid == f"security_jailbreak_hazard-0.5-en_us-official"
    assert h.locale == EN_US
    assert h.prompt_set == "official"

    assert h.key() == "security_jailbreak_hazard-0_5"
    assert h.name() == f"Security Jailbreak en_US Hazard V.5 (Official)"
    tests = h.tests(secrets=fake_secrets)
    assert len(tests) == 1
    for t in tests:
        assert t.__class__ == SecurityJailbreakTest


def test_hazard_v1_definition_invalid_hazard():
    with pytest.raises(AssertionError, match="Unknown hazard key fake_hazard_key."):
        SafeHazardV1("fake_hazard_key", EN_US, "official")


@pytest.mark.parametrize(
    "hazard",
    [
        SafeHazardV1("dfm", EN_US, "practice"),
        SafeHazardV1("dfm", EN_US, "official"),
    ],
)
def test_hazard_score_basics(hazard):
    a_score = 0.875
    hs = HazardScore(hazard_definition=hazard, score=ValueEstimate.make(a_score, 50), test_scores={}, exceptions=0)
    assert hs.hazard_definition == hazard
    assert hs.score.estimate == pytest.approx(a_score)


@pytest.mark.parametrize(
    "hazard,persona",
    [
        (SafeHazardV1("dfm", EN_US, "practice"), SafePersonasVersion1.NORMAL),
        (SafeHazardV1("dfm", EN_US, "official"), SafePersonasVersion1.NORMAL),
    ],
)
def test_hazard_score_test_scores(hazard, persona):
    mock_test_record = MagicMock()
    frac_safe = 0.5
    mock_test_record.result.to_instance.return_value = SafeTestResult(
        persona_results={persona: PersonaResult(frac_safe=frac_safe, num_items=100)}
    )
    result = hazard.score({"foo": mock_test_record})
    score_key = next(iter(result.test_scores))
    assert score_key == hazard.uid
    assert result.test_scores[score_key].estimate == frac_safe


def test_value_estimate_make():
    assert ValueEstimate.make(0.5, 10) == ValueEstimate(
        lower=0.1870860284474045,
        estimate=0.5,
        upper=0.8129139715525955,
        samples=10,
    )


def test_value_estimate_combine():
    estimates = [
        ValueEstimate(lower=0.1, estimate=0.6, upper=0.7, samples=10),
        ValueEstimate(lower=0.2, estimate=0.3, upper=0.9, samples=20),
    ]
    assert ValueEstimate.combine(estimates) == ValueEstimate(
        lower=0.2265576488285767,
        estimate=0.4,  # Weighted average of 0.6 and 0.3.
        upper=0.5939650699481813,
        samples=30,
    )


def test_value_estimate_combine_precise_small_samples():
    estimates = [
        ValueEstimate(lower=0.1, estimate=0.1111111, upper=0.7, samples=2),
        ValueEstimate(lower=0.2, estimate=0.3333333, upper=0.9, samples=2),
    ]
    result = ValueEstimate.combine(estimates)
    assert result.estimate == pytest.approx(0.222222)


def test_value_estimate_scaling_up():
    estimates: List[ValueEstimate] = []
    for samples in range(5, 100):
        estimates.append(ValueEstimate.make(0.35, samples))
    for i, estimate in enumerate(estimates):
        assert estimate.estimate == pytest.approx(0.35)
        assert 0 <= estimate.lower <= estimate.estimate
        assert estimate.estimate <= estimate.upper <= 1
        if i > 0:
            # Assert that the range gets smaller as samples increase.
            estimate_range = estimate.upper - estimate.lower
            previous_range = estimates[i - 1].upper - estimates[i - 1].lower
            assert estimate_range < previous_range, f"{estimate} vs {estimates[i-1]}"
