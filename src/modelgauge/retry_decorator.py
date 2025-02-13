import time
import functools

BASE_RETRY_COUNT = 3
MAX_RETRY_DURATION = 86400  # 1 day in seconds
MAX_BACKOFF = 60  # 1 minute in seconds


def retry(unacceptable_exceptions=None):
    """
    A decorator that retries a function at least BASE_RETRY_COUNT times.
    If unacceptable_exceptions are specified, it will retry for up to 1 day if any of those exceptions occur.
    """
    unacceptable_exceptions = tuple(unacceptable_exceptions) if unacceptable_exceptions else ()

    # TODO: Is functools.wraps necessary? Test if journal works without it.
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            start_time = time.time()

            while True:
                try:
                    return func(*args, **kwargs)
                except unacceptable_exceptions as e:
                    # Keep retrying "unacceptable" exceptions for 1 day.
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= MAX_RETRY_DURATION:
                        raise
                except:
                    # Retry all other exceptions BASE_RETRY_COUNT times.
                    attempt += 1
                    if attempt >= BASE_RETRY_COUNT:
                        raise
                sleep_time = min(2**attempt, MAX_BACKOFF)  # Exponential backoff with cap
                time.sleep(sleep_time)

        return wrapper

    return decorator
