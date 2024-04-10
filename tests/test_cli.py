import os
import pytest
from tests.utilities import expensive_tests


@expensive_tests
def test_main():
    assert os.system("modelgauge") == 0


@expensive_tests
def test_list_plugins():
    assert os.system("modelgauge list") == 0


@expensive_tests
def test_list_secrets():
    assert os.system("modelgauge list-secrets") == 0


@expensive_tests
def test_list_tests():
    assert os.system("modelgauge list-tests") == 0


@expensive_tests
def test_list_suts():
    assert os.system("modelgauge list-suts") == 0


@expensive_tests
@pytest.mark.parametrize(
    "sut",
    [
        "demo_yes_no",
        "demo_random_words",
        "demo_always_angry",
        "demo_always_sorry",
    ],
)
def test_run_sut_demos(sut):
    assert (
        os.system(
            f"""modelgauge run-sut \
                --sut {sut} \
                --prompt "Can you say Hello?" """
        )
        == 0
    )


@expensive_tests
@pytest.mark.parametrize("test", ["demo_01", "demo_02", "demo_03", "demo_04"])
def test_run_test_demos(test):
    assert (
        os.system(
            f"""modelgauge run-test \
                --test {test} \
                --sut demo_yes_no \
                --max-test-items 1"""
        )
        == 0
    )
