import functools
import time

BASE_RETRY_COUNT = 3
MAX_RETRY_DURATION = 86400  # 1 day in seconds
MAX_BACKOFF = 60  # 1 minute in seconds


def retry(
    transient_exceptions=None,
    base_retry_count=BASE_RETRY_COUNT,
    max_retry_duration=MAX_RETRY_DURATION,
    max_backoff=MAX_BACKOFF,
):
    """
    A decorator that retries a function at least base_retry_count times.
    If transient_exceptions are specified, it will retry for up to 1 day if any of those exceptions occur.
    """
    transient_exceptions = tuple(transient_exceptions) if transient_exceptions else ()

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            start_time = time.time()

            while True:
                try:
                    return func(*args, **kwargs)
                except transient_exceptions as e:
                    # Keep retrying transient exceptions for 1 day.
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= max_retry_duration:
                        raise
                except:
                    # Retry all other exceptions BASE_RETRY_COUNT times.
                    attempt += 1
                    if attempt >= base_retry_count:
                        raise
                sleep_time = min(2**attempt, max_backoff)  # Exponential backoff with cap
                time.sleep(sleep_time)

        return wrapper

    return decorator
