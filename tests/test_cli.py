import os
import pathlib

import pytest

from tests.utilities import expensive_tests


@pytest.fixture
def cmd():
    return pathlib.Path(__file__).parent.parent / "newhelm" / "main.py"


@expensive_tests
def test_main(cmd):
    assert os.system(f"python {cmd}") == 0


@expensive_tests
def test_list_plugins(cmd):
    assert os.system(f"python {cmd} list") == 0


@expensive_tests
@pytest.mark.parametrize("sut", ["DemoMultipleChoiceSUT"])
@pytest.mark.parametrize("test", ["demo_01", "demo_02"])
def test_test_sut_combinations(cmd, test, sut):
    assert (
        os.system(
            f"""python {cmd} run-test \
                --test {test} \
                --sut {sut} \
                --max-test-items 1"""
        )
        == 0
    )
