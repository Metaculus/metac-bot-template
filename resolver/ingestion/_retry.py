"""Generic retry helper for ingestion connectors."""

from __future__ import annotations

import inspect
import logging
import random
import subprocess
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
            exit_code = getattr(exc, "returncode", None)
            stderr_text = ""
            stdout_text = ""
            try:
                if hasattr(exc, "stderr") and getattr(exc, "stderr"):
                    raw_stderr = getattr(exc, "stderr")
                    if isinstance(raw_stderr, (bytes, bytearray)):
                        stderr_text = raw_stderr.decode(errors="ignore")
                    else:
                        stderr_text = str(raw_stderr)
                if hasattr(exc, "stdout") and getattr(exc, "stdout"):
                    raw_stdout = getattr(exc, "stdout")
                    if isinstance(raw_stdout, (bytes, bytearray)):
                        stdout_text = raw_stdout.decode(errors="ignore")
                    else:
                        stdout_text = str(raw_stdout)
            except Exception:  # noqa: BLE001
                stderr_text = ""
                stdout_text = ""
            if isinstance(exc, subprocess.CalledProcessError):
                exit_code = exc.returncode

            should_retry = attempt <= attempts_allowed and condition(exc)
            backoff = 0.0
            if should_retry and attempt <= attempts_allowed:
                backoff = min(max_delay, base_delay * (2 ** (attempt - 1)))
                if jitter:
                    backoff += random.uniform(0, backoff * 0.333)
            if attempt_logger is not None:
                payload = {
                    "event": "retry",
                    "attempt": attempt,
                    "connector": connector,
                    "retry": should_retry,
                    "sleep": round(backoff, 3),
                }
                if exit_code is not None:
                    payload["exit_code"] = exit_code
                if stderr_text:
                    payload["stderr_excerpt"] = stderr_text[:200]
                if stdout_text:
                    payload["stdout_excerpt"] = stdout_text[:200]
                attempt_logger.warning("attempt failed", exc_info=exc, extra=payload)
            if not should_retry:
                raise
            if backoff > 0:
                time.sleep(backoff)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("retry_call exhausted without executing the callable")
