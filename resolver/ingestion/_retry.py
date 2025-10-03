"""Generic retry helper for ingestion connectors."""

from __future__ import annotations

import inspect
import logging
import random
import time
from typing import Any, Callable, Optional

RetryFunc = Callable[..., Any]
RetryCondition = Callable[[BaseException], bool]


_DEFAULT_RETRYABLE_NAMES = (
    "Timeout",
    "ConnectionError",
    "SSLError",
    "ReadTimeout",
    "ChunkedEncodingError",
)

_DEFAULT_RETRYABLE_PHRASES = ("rate limit", "waf", "temporarily unavailable")


def _default_is_retryable(exc: BaseException) -> bool:
    exc_name = exc.__class__.__name__
    if any(name in exc_name for name in _DEFAULT_RETRYABLE_NAMES):
        return True
    message = str(exc).lower()
    return any(phrase in message for phrase in _DEFAULT_RETRYABLE_PHRASES)


def retry_call(
    func: RetryFunc,
    *,
    retries: int = 2,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: bool = True,
    is_retryable: Optional[RetryCondition] = None,
    logger: Optional[logging.Logger | logging.LoggerAdapter] = None,
    connector: Optional[str] = None,
):
    """Invoke ``func`` with retry and exponential backoff."""

    condition = is_retryable or _default_is_retryable
    attempts_allowed = max(0, retries)

    signature = inspect.signature(func)
    takes_attempt = len(signature.parameters) >= 1

    last_exc: BaseException | None = None

    for attempt in range(1, attempts_allowed + 2):
        attempt_logger = logger
        if isinstance(logger, logging.LoggerAdapter):
            attempt_logger = logger
        try:
            if takes_attempt:
                result = func(attempt)
            else:
                result = func()
            return result
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            should_retry = attempt <= attempts_allowed and condition(exc)
            sleep_for = min(max_delay, base_delay * (2 ** (attempt - 1)))
            if jitter:
                sleep_for += random.uniform(0, sleep_for * 0.333)
            if attempt_logger is not None:
                payload = {
                    "event": "retry",
                    "attempt": attempt,
                    "connector": connector,
                    "retry": should_retry,
                    "sleep": round(sleep_for, 3),
                }
                attempt_logger.warning("attempt failed", exc_info=exc, extra=payload)
            if not should_retry:
                raise
            time.sleep(sleep_for)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("retry_call exhausted without executing the callable")
