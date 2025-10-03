"""Logging utilities for the ingestion runner."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

DEFAULT_LOG_DIR = Path("resolver/logs/ingestion")


@dataclass(frozen=True)
class _RunnerContext:
    run_id: str
    log_dir: Path


class _RunnerFormatter(logging.Formatter):
    """Formatter that ensures UTC timestamps and runner metadata."""

    converter = time.gmtime

    def __init__(self, context: _RunnerContext) -> None:
        super().__init__(
            fmt="%(asctime)sZ [%(levelname)s] [run:%(run_id)s] [connector:%(connector)s] "
            "attempt=%(attempt)s %(message)s"
        )
        self._context = context

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        _apply_defaults(record, self._context)
        return super().format(record)


class _ConsoleJSONFormatter(logging.Formatter):
    """Formatter that renders log records as JSON for stdout."""

    converter = time.gmtime

    def __init__(self, context: _RunnerContext) -> None:
        super().__init__()
        self._context = context

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        _apply_defaults(record, self._context)
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "run_id": record.run_id,
            "connector": record.connector,
            "attempt": record.attempt,
            "message": redact(record.getMessage()),
        }
        if record.__dict__:
            extras = {
                key: value
                for key, value in record.__dict__.items()
                if key not in {"run_id", "connector", "attempt", "msg", "args"}
            }
            if extras:
                payload["extra"] = _jsonify(extras)
        if record.exc_info:
            payload["exc"] = redact("".join(traceback.format_exception(*record.exc_info)))
        return json.dumps(payload, ensure_ascii=False)


class _JSONLineHandler(logging.Handler):
    """Write structured log records to a JSONL file."""

    def __init__(self, path: Path, context: _RunnerContext) -> None:
        super().__init__()
        self._path = path
        self._context = context
        path.parent.mkdir(parents=True, exist_ok=True)
        # Open in text mode with UTF-8 encoding so json is readable.
        self._stream = path.open("a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            payload = self._build_payload(record)
            json.dump(payload, self._stream, ensure_ascii=False, separators=(",", ":"))
            self._stream.write("\n")
            self._stream.flush()
        except Exception:
            self.handleError(record)

    def _build_payload(self, record: logging.LogRecord) -> Dict[str, Any]:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
        payload: Dict[str, Any] = {
            "ts": f"{ts}.{int(record.msecs):03d}Z",
            "level": record.levelname,
            "run_id": getattr(record, "run_id", self._context.run_id),
            "connector": getattr(record, "connector", "-"),
            "attempt": getattr(record, "attempt", 0),
            "msg": redact(record.getMessage()),
        }

        extra = {
            key: value
            for key, value in record.__dict__.items()
            if key
            not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "run_id",
                "connector",
                "attempt",
                "message",
                "duration_ms",
            }
        }
        if extra:
            payload["extra"] = _jsonify(extra)

        if record.exc_info:
            payload["exc"] = redact("".join(traceback.format_exception(*record.exc_info)))
        return payload

    def close(self) -> None:  # noqa: D401
        try:
            self._stream.close()
        finally:
            super().close()


def _jsonify(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return redact(value) if isinstance(value, str) else value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(v) for v in value]
    return redact(repr(value))


_BASE_LOGGER: logging.Logger | None = None
_CONTEXT: _RunnerContext | None = None
_SENSITIVE_PATTERN = re.compile(r"[A-Za-z0-9_\-]{24,}")
_SENSITIVE_VALUES: set[str] = set()


def _apply_defaults(record: logging.LogRecord, context: _RunnerContext) -> None:
    if not getattr(record, "run_id", None):
        record.run_id = context.run_id  # type: ignore[attr-defined]
    if getattr(record, "connector", None) in {None, ""}:
        record.connector = "-"  # type: ignore[attr-defined]
    if getattr(record, "attempt", None) in {None, ""}:
        record.attempt = 0  # type: ignore[attr-defined]
    if getattr(record, "duration_ms", None) in {None, ""}:
        record.duration_ms = 0  # type: ignore[attr-defined]


def _capture_sensitive_env() -> None:
    global _SENSITIVE_VALUES
    candidates = []
    for key, value in os.environ.items():
        upper = key.upper()
        if any(token in upper for token in ("TOKEN", "KEY", "SECRET", "PASSWORD")) and value:
            candidates.append(value)
    _SENSITIVE_VALUES = {val for val in candidates if len(val) >= 8}


def init_logger(
    run_id: str,
    level: str | None = None,
    fmt: str | None = None,
    log_dir: Path | None = None,
) -> logging.Logger:
    """Initialise the structured logger for ingestion runs."""

    global _BASE_LOGGER, _CONTEXT

    env_level = os.environ.get("RUNNER_LOG_LEVEL", "INFO")
    env_format = os.environ.get("RUNNER_LOG_FORMAT", "plain")
    env_dir = os.environ.get("RUNNER_LOG_DIR")

    log_dir = Path(env_dir) if env_dir else (log_dir or DEFAULT_LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    chosen_level = (level or env_level).upper()
    level_value = getattr(logging, chosen_level, logging.INFO)
    chosen_format = (fmt or env_format).lower()

    context = _RunnerContext(run_id=run_id, log_dir=log_dir)

    logger = logging.getLogger("resolver.ingestion.runner")
    logger.handlers.clear()
    logger.setLevel(level_value)
    logger.propagate = False

    plain_formatter = _RunnerFormatter(context)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level_value)
    if chosen_format == "json":
        console_handler.setFormatter(_ConsoleJSONFormatter(context))
    else:
        console_handler.setFormatter(plain_formatter)
    logger.addHandler(console_handler)

    text_log = log_dir / f"ingest_{run_id}.log"
    file_handler = logging.FileHandler(text_log, encoding="utf-8")
    file_handler.setLevel(level_value)
    file_handler.setFormatter(plain_formatter)
    logger.addHandler(file_handler)

    json_log = log_dir / f"ingest_{run_id}.jsonl"
    json_handler = _JSONLineHandler(json_log, context)
    json_handler.setLevel(level_value)
    logger.addHandler(json_handler)

    logger.run_id = run_id  # type: ignore[attr-defined]
    logger.log_dir = log_dir  # type: ignore[attr-defined]
    logger.json_log_path = json_log  # type: ignore[attr-defined]
    logger.text_log_path = text_log  # type: ignore[attr-defined]

    _BASE_LOGGER = logger
    _CONTEXT = context
    _capture_sensitive_env()
    return logger


def child_logger(connector_name: str) -> logging.LoggerAdapter:
    if _BASE_LOGGER is None:
        raise RuntimeError("init_logger must be called before child_logger")
    return logging.LoggerAdapter(_BASE_LOGGER, {"connector": connector_name})


def log_env_summary(logger: logging.Logger) -> None:
    """Log runtime environment details to help with debugging."""

    git_rev = _run_git(["rev-parse", "--short", "HEAD"])
    git_branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    python_version = sys.version.replace("\n", " ")
    platform = sys.platform
    tokens = {
        key: ("yes" if os.environ.get(key) else "no")
        for key in sorted(os.environ)
        if any(token in key.upper() for token in ("TOKEN", "KEY", "SECRET", "PASSWORD"))
    }
    summary = {
        "event": "env_summary",
        "git": {"rev": git_rev, "branch": git_branch},
        "python": python_version,
        "platform": platform,
        "resolver_ci": os.environ.get("RESOLVER_CI", ""),
        "disable_git_push": os.environ.get("DISABLE_GIT_PUSH", ""),
        "tokens": tokens,
    }
    logger.info("environment", extra=summary)


def _run_git(args: Iterable[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )
        output = result.stdout.strip()
        return output if output else "unknown"
    except Exception:
        return "unavailable"


def redact(value: str) -> str:
    if not isinstance(value, str):
        return value
    masked = _SENSITIVE_PATTERN.sub("***", value)
    for secret in _SENSITIVE_VALUES:
        if secret:
            masked = masked.replace(secret, "***")
    return masked


def connector_log_path(connector: str) -> Path:
    if _CONTEXT is None:
        raise RuntimeError("init_logger must be called before connector_log_path")
    safe = connector.replace("/", "_")
    path = _CONTEXT.log_dir / _CONTEXT.run_id / f"{safe}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def attach_connector_handler(
    logger: logging.LoggerAdapter | logging.Logger, connector: str
) -> logging.Handler:
    if _CONTEXT is None:
        raise RuntimeError("init_logger must be called before attaching handlers")
    base_logger = logger.logger if isinstance(logger, logging.LoggerAdapter) else logger
    handler = logging.FileHandler(connector_log_path(connector), encoding="utf-8")
    handler.setFormatter(_RunnerFormatter(_CONTEXT))
    base_logger.addHandler(handler)
    return handler


def detach_connector_handler(
    logger: logging.LoggerAdapter | logging.Logger, handler: logging.Handler
) -> None:
    base_logger = logger.logger if isinstance(logger, logging.LoggerAdapter) else logger
    base_logger.removeHandler(handler)
    handler.close()
