import functools
import time

from modellogger.log_config import get_logger

BASE_RETRY_COUNT = 3
MAX_RETRY_DURATION = 86400  # 1 day in seconds
MAX_BACKOFF = 60  # 1 minute in seconds

logger = get_logger(__name__)


def retry(
    do_not_retry_exceptions=None,
    transient_exceptions=None,
    base_retry_count=BASE_RETRY_COUNT,
    max_retry_duration=MAX_RETRY_DURATION,
    max_backoff=MAX_BACKOFF,
):
    """
    A decorator that retries a function at least base_retry_count times.
    If do_not_retry_exceptions are specified, it will not retry if any of those exceptions occur.
    If transient_exceptions are specified, it will retry for up to 1 day if any of those exceptions occur.
    """
    do_not_retry_exceptions = tuple(do_not_retry_exceptions) if do_not_retry_exceptions else ()
    transient_exceptions = tuple(transient_exceptions) if transient_exceptions else ()

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            start_time = time.time()

            while True:
                try:
                    return func(*args, **kwargs)
                except do_not_retry_exceptions as e:
                    raise
                except transient_exceptions as e:
                    # Keep retrying transient exceptions for 1 day.
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= max_retry_duration:
                        raise
                    logger.warning(f"Transient exception occurred: {e}. Retrying...")
                except Exception as e:
                    # Retry all other exceptions BASE_RETRY_COUNT times.
                    attempt += 1
                    if attempt >= base_retry_count:
                        raise
                    logger.warning(f"Exception occurred after {attempt}/{base_retry_count} attempts: {e}. Retrying...")
                sleep_time = min(2**attempt, max_backoff)  # Exponential backoff with cap
                time.sleep(sleep_time)

        return wrapper

    return decorator
