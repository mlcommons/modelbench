import pytest
import time

from modelgauge.retry_decorator import retry, BASE_RETRY_COUNT


def test_retry_success():
    @retry()
    def always_succeed():
        return "success"

    assert always_succeed() == "success"


@pytest.mark.parametrize("exceptions", [None, [ValueError]])
def test_retry_fails_after_base_retries(exceptions):
    attempt_counter = 0

    @retry(unacceptable_exceptions=exceptions)
    def always_fail():
        nonlocal attempt_counter
        attempt_counter += 1
        raise KeyError("Intentional failure")

    with pytest.raises(KeyError):
        always_fail()

    assert attempt_counter == BASE_RETRY_COUNT


def test_retry_eventually_succeeds():
    attempt_counter = 0

    @retry(unacceptable_exceptions=[ValueError])
    def succeed_after_two_attempts():
        nonlocal attempt_counter
        attempt_counter += 1
        if attempt_counter < BASE_RETRY_COUNT:
            raise ValueError("Intentional failure")
        return "success"

    assert succeed_after_two_attempts() == "success"
    assert attempt_counter == BASE_RETRY_COUNT


def test_retry_unacceptable_eventually_succeeds():
    attempt_counter = 0
    start_time = time.time()

    @retry(unacceptable_exceptions=[ValueError])
    def succeed_eventually():
        nonlocal attempt_counter
        attempt_counter += 1
        elapsed_time = time.time() - start_time
        if elapsed_time < 5:
            raise ValueError("Intentional failure")
        return "success"

    assert succeed_eventually() == "success"
    assert attempt_counter > BASE_RETRY_COUNT
