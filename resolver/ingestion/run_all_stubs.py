#!/usr/bin/env python3
"""Run ingestion connectors and optional stubs to populate staging CSVs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import os
import re
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from resolver.ingestion._retry import retry_call
from resolver.ingestion._runner_logging import (
    attach_connector_handler,
    child_logger,
    detach_connector_handler,
    init_logger,
    log_env_summary,
    redact,
)

STAGING = ROOT.parent / "staging"
CONFIG_DIR = ROOT / "config"
LOGS_DIR = ROOT.parent / "logs" / "ingestion"
RESOLVER_DEBUG = bool(int(os.getenv("RESOLVER_DEBUG", "0") or 0))


def _repo_root() -> Path:
    """Return the repository root directory."""

    return Path(__file__).resolve().parents[2]


def _module_name_from_path(py_path: Path) -> str:
    """Convert a Python file path to its dotted module path."""

    if not py_path.exists():
        raise FileNotFoundError(f"Connector file does not exist: {py_path}")

    no_ext = py_path.with_suffix("")
    parts = list(no_ext.parts)
    try:
        idx = parts.index("resolver")
    except ValueError as exc:
        raise RuntimeError(
            f"Connector path '{py_path}' does not include 'resolver' package root."
        ) from exc

    module = ".".join(parts[idx:])
    if not module:
        raise RuntimeError(f"Failed to derive module name from {py_path}")
    return module

INGESTION_MODE = (os.environ.get("RESOLVER_INGESTION_MODE") or "").strip().lower()
INCLUDE_STUBS = os.environ.get("RESOLVER_INCLUDE_STUBS", "0") == "1"
FAIL_ON_STUB_ERROR = os.environ.get("RESOLVER_FAIL_ON_STUB_ERROR", "0") == "1"
FORCE_DTM_STUB = os.environ.get("RESOLVER_FORCE_DTM_STUB", "0") == "1"

REAL = [
    "ifrc_go_client.py",
    "reliefweb_client.py",
    "unhcr_client.py",
    "unhcr_odp_client.py",
    "who_phe_client.py",
    "ipc_client.py",
    "wfp_mvam_client.py",
    "acled_client.py",
    "dtm_client.py",
    "hdx_client.py",
    "emdat_client.py",
    "gdacs_client.py",
    "worldpop_client.py",
]

SUMMARY_TARGETS = {
    "who_phe_client.py": {
        "label": "WHO-PHE",
        "staging": STAGING / "who_phe.csv",
        "config": CONFIG_DIR / "who_phe.yml",
    },
    "wfp_mvam_client.py": {
        "label": "WFP-mVAM",
        "staging": STAGING / "wfp_mvam.csv",
        "config": CONFIG_DIR / "wfp_mvam_sources.yml",
    },
    "ipc_client.py": {
        "label": "IPC",
        "staging": STAGING / "ipc.csv",
        "config": CONFIG_DIR / "ipc.yml",
    },
    "unhcr_client.py": {
        "label": "UNHCR",
        "staging": STAGING / "unhcr.csv",
        "config": CONFIG_DIR / "unhcr.yml",
    },
    "unhcr_odp_client.py": {
        "label": "UNHCR-ODP",
        "staging": STAGING / "unhcr_odp.csv",
        "config": None,
    },
    "acled_client.py": {
        "label": "ACLED",
        "staging": STAGING / "acled.csv",
        "config": CONFIG_DIR / "acled.yml",
    },
    "dtm_client.py": {
        "label": "DTM",
        "staging": STAGING / "dtm.csv",
        "config": CONFIG_DIR / "dtm.yml",
    },
}

STUBS = [
    "ifrc_go_stub.py",
    "reliefweb_stub.py",
    "unhcr_stub.py",
    "hdx_stub.py",
    "who_stub.py",
    "ipc_stub.py",
    "emdat_stub.py",
    "gdacs_stub.py",
    "copernicus_stub.py",
    "unosat_stub.py",
    "acled_stub.py",
    "ucdp_stub.py",
    "fews_stub.py",
    "wfp_mvam_stub.py",
    "gov_ndma_stub.py",
]

if FORCE_DTM_STUB:
    REAL = [name for name in REAL if name != "dtm_client.py"]
    if "dtm_stub.py" not in STUBS:
        STUBS.insert(0, "dtm_stub.py")
else:
    STUBS = [name for name in STUBS if name != "dtm_stub.py"]

SKIP_ENVS = {
    "ifrc_go_client.py": ("RESOLVER_SKIP_IFRCGO", "IFRC GO connector"),
    "reliefweb_client.py": ("RESOLVER_SKIP_RELIEFWEB", "ReliefWeb connector"),
    "unhcr_client.py": ("RESOLVER_SKIP_UNHCR", "UNHCR connector"),
    "unhcr_odp_client.py": ("RESOLVER_SKIP_UNHCR_ODP", "UNHCR ODP connector"),
    "acled_client.py": ("RESOLVER_SKIP_ACLED", "ACLED connector"),
    "dtm_client.py": ("RESOLVER_SKIP_DTM", "DTM connector"),
    "emdat_client.py": ("RESOLVER_SKIP_EMDAT", "EM-DAT connector"),
    "gdacs_client.py": ("RESOLVER_SKIP_GDACS", "GDACS connector"),
    "who_phe_client.py": ("RESOLVER_SKIP_WHO", "WHO PHE connector"),
    "ipc_client.py": ("RESOLVER_SKIP_IPC", "IPC connector"),
    "hdx_client.py": ("RESOLVER_SKIP_HDX", "HDX connector"),
    "worldpop_client.py": ("RESOLVER_SKIP_WORLDPOP", "WorldPop connector"),
    "wfp_mvam_client.py": ("RESOLVER_SKIP_WFP_MVAM", "WFP mVAM connector"),
}


# Prefer known "fatal" / non-retryable exit codes (sysexits) when available
_EX_USAGE = getattr(os, "EX_USAGE", 64)
_EX_DATAERR = getattr(os, "EX_DATAERR", 65)
_EX_NOINPUT = getattr(os, "EX_NOINPUT", 66)
_EX_NOUSER = getattr(os, "EX_NOUSER", 67)
_EX_NOHOST = getattr(os, "EX_NOHOST", 68)
_EX_UNAVAILABLE = getattr(os, "EX_UNAVAILABLE", 69)
_EX_SOFTWARE = getattr(os, "EX_SOFTWARE", 70)
_EX_OSERR = getattr(os, "EX_OSERR", 71)
_EX_OSFILE = getattr(os, "EX_OSFILE", 72)
_EX_CANTCREAT = getattr(os, "EX_CANTCREAT", 73)
_EX_IOERR = getattr(os, "EX_IOERR", 74)
_EX_TEMPFAIL = getattr(os, "EX_TEMPFAIL", 75)
_EX_PROTOCOL = getattr(os, "EX_PROTOCOL", 76)
_EX_NOPERM = getattr(os, "EX_NOPERM", 77)
_EX_CONFIG = getattr(os, "EX_CONFIG", 78)

# Treat these as non-retryable (usage/config/software errors)
NON_RETRYABLE_EXIT_CODES = {
    2,  # bad CLI usage (common)
    _EX_USAGE,
    _EX_DATAERR,
    _EX_NOINPUT,
    _EX_NOUSER,
    _EX_NOHOST,
    _EX_UNAVAILABLE,
    _EX_SOFTWARE,
    _EX_OSERR,
    _EX_OSFILE,
    _EX_CANTCREAT,
    _EX_IOERR,
    _EX_PROTOCOL,
    _EX_NOPERM,
    _EX_CONFIG,
}

# Heuristics for transient errors in stderr/stdout
_TRANSIENT_PAT = re.compile(
    r"(timed out|timeout|temporar\w+ unavailable|connection reset|"
    r"connection aborted|connection refused|network is unreachable|"
    r"ECONNRESET|ETIMEDOUT|EHOSTUNREACH|ENETUNREACH|EAI_AGAIN|"
    r"rate limit|429|5\d{2}\b|service unavailable|try again)",
    re.IGNORECASE,
)


def _coerce_process_stream(stream: object | None) -> str:
    if stream is None:
        return ""
    if isinstance(stream, (bytes, bytearray)):
        try:
            return stream.decode(errors="ignore")
        except Exception:  # noqa: BLE001
            return ""
    return str(stream)


def _is_retryable_exception(
    exc: BaseException,
    *,
    exit_code: int | None = None,
    stderr: str | None = None,
    stdout: str | None = None,
) -> bool:
    # Network/timeouts raised as exceptions by helpers remain retryable
    transient_types = (TimeoutError, ConnectionError)
    if isinstance(exc, transient_types):
        return True

    # Subprocess script failures: retry unless clearly non-retryable
    if isinstance(exc, subprocess.CalledProcessError):
        code = exc.returncode if exit_code is None else exit_code
        if code in NON_RETRYABLE_EXIT_CODES:
            return False
        text_parts: list[str] = []
        if stderr is not None:
            text_parts.append(stderr)
        if stdout is not None:
            text_parts.append(stdout)
        if not text_parts:
            if hasattr(exc, "stderr") and exc.stderr:
                text_parts.append(_coerce_process_stream(exc.stderr))
            if hasattr(exc, "stdout") and exc.stdout:
                text_parts.append(_coerce_process_stream(exc.stdout))
        text = "\n".join(part for part in text_parts if part)
        # Retry on unknown codes if we see transient hints; otherwise still retry (assume transient)
        return True if not text else bool(_TRANSIENT_PAT.search(text))

    # Default to non-retryable
    exc_name = exc.__class__.__name__
    transient_names = (
        "Timeout",
        "ConnectionError",
        "SSLError",
        "ReadTimeout",
        "ChunkedEncodingError",
    )
    if any(name in exc_name for name in transient_names):
        return True
    message = str(exc).lower()
    transient_phrases = ("rate limit", "waf", "temporarily unavailable")
    return any(phrase in message for phrase in transient_phrases)


@dataclass
class ConnectorSpec:
    filename: str
    path: Path
    kind: str
    output_path: Optional[Path] = None
    summary: Optional[str] = None
    skip_reason: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.filename.rsplit(".", 1)[0]


def _load_yaml(path: Optional[Path]) -> dict:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _count_rows(path: Optional[Path]) -> int:
    if not path or not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            return sum(1 for _ in reader)
    except Exception:
        return 0


def _summarise_connector(name: str) -> str | None:
    meta = SUMMARY_TARGETS.get(name)
    if not meta:
        return None
    label = meta["label"]
    rows = _count_rows(meta["staging"])
    parts = [f"[{label}] rows:{rows}"]
    cfg = _load_yaml(meta.get("config"))

    if name == "who_phe_client.py":
        enabled = bool(cfg.get("enabled", False))
        sources_cfg = cfg.get("sources", {})
        configured = 0
        if isinstance(sources_cfg, dict):
            for value in sources_cfg.values():
                if isinstance(value, dict):
                    url = str(value.get("url", "")).strip()
                else:
                    url = str(value).strip()
                if url:
                    configured += 1
        parts.append(f"enabled:{'yes' if enabled else 'no'}")
        parts.append(f"sources:{configured}")
    elif name == "wfp_mvam_client.py":
        enabled = bool(cfg.get("enabled", False))
        sources = cfg.get("sources", [])
        count = 0
        if isinstance(sources, dict):
            sources = list(sources.values())
        if isinstance(sources, list):
            for entry in sources:
                if isinstance(entry, dict) and str(entry.get("url", "")).strip():
                    count += 1
                elif isinstance(entry, str) and entry.strip():
                    count += 1
        parts.append(f"enabled:{'yes' if enabled else 'no'}")
        parts.append(f"sources:{count}")
    elif name == "ipc_client.py":
        enabled = bool(cfg.get("enabled", False))
        feeds = cfg.get("feeds", [])
        if isinstance(feeds, dict):
            feed_count = sum(1 for value in feeds.values() if value)
        elif isinstance(feeds, list):
            feed_count = len(feeds)
        else:
            feed_count = 0
        parts.append(f"enabled:{'yes' if enabled else 'no'}")
        parts.append(f"feeds:{feed_count}")
    elif name == "unhcr_client.py":
        years = cfg.get("include_years") or []
        years_text: str
        if isinstance(years, list) and years:
            years_text = ",".join(str(y) for y in years)
        else:
            try:
                years_back = int(cfg.get("years_back", 3) or 0)
            except Exception:
                years_back = 3
            current = dt.date.today().year
            years_text = ",".join(str(current - offset) for offset in range(years_back + 1))
        parts.append(f"years:{years_text}")
    elif name == "acled_client.py":
        token_present = bool(os.getenv("ACLED_TOKEN") or str(cfg.get("token", "")).strip())
        parts.append(f"token:{'yes' if token_present else 'no'}")
    return " ".join(parts)


def _safe_summary(name: str) -> str | None:
    try:
        summary = _summarise_connector(name)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning(
            "failed to summarise connector %s", name, exc_info=exc
        )
        return None
    if summary:
        print(summary)
    return summary


def _should_skip(script: str) -> Optional[str]:
    env_name, label = SKIP_ENVS.get(script, (None, None))
    if env_name and os.environ.get(env_name) == "1":
        return f"{env_name}=1 — {label}"
    return None


def _normalise_name(name: str) -> str:
    name = name.strip()
    if not name:
        return name
    if not name.endswith(".py"):
        name = f"{name}.py"
    return name


def _build_specs(
    real: Sequence[str],
    stubs: Sequence[str],
    selected: set[str] | None,
    run_real: bool,
    run_stubs: bool,
) -> List[ConnectorSpec]:
    specs: List[ConnectorSpec] = []
    if run_real:
        for filename in real:
            if selected and filename not in selected:
                continue
            specs.append(_create_spec(filename, "real"))
    if run_stubs:
        for filename in stubs:
            if selected and filename not in selected:
                continue
            specs.append(_create_spec(filename, "stub"))
    return specs


def _create_spec(filename: str, kind: str) -> ConnectorSpec:
    path = ROOT / filename
    meta = SUMMARY_TARGETS.get(filename, {})
    summary = _safe_summary(filename)
    output_path = meta.get("staging") if isinstance(meta, dict) else None
    skip_reason = None
    skip_env = _should_skip(filename)
    if not path.exists():
        skip_reason = f"missing: {filename}"
    elif skip_env:
        skip_reason = skip_env
    metadata: Dict[str, str] = {}
    if output_path:
        metadata["output_path"] = str(output_path)
    return ConnectorSpec(
        filename=filename,
        path=path,
        kind=kind,
        output_path=output_path,
        summary=summary,
        skip_reason=skip_reason,
        metadata=metadata,
    )


def _invoke_connector(path: Path, *, logger: logging.Logger | logging.LoggerAdapter | None = None) -> None:
    repo_root = _repo_root()
    module = _module_name_from_path(path)
    log = logger if logger is not None else logging.getLogger(__name__)
    log.info(
        "launching connector",
        extra={"event": "launch", "module": module, "cwd": str(repo_root)},
    )
    cmd = [sys.executable, "-m", module]
    try:
        proc = subprocess.run(cmd, cwd=repo_root)
    except OSError as exc:  # pragma: no cover - defensive
        log.error(
            "failed to start connector",
            extra={"event": "launch_error", "module": module, "error": str(exc)},
        )
        raise RuntimeError(f"{module} failed to start: {exc}") from exc
    if proc.returncode != 0:
        log.error(
            "connector exited with non-zero status",
            extra={
                "event": "launch_failed",
                "module": module,
                "returncode": proc.returncode,
            },
        )
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def _with_attempt_logger(
    base: logging.LoggerAdapter | logging.Logger, attempt: int
) -> logging.LoggerAdapter:
    if isinstance(base, logging.LoggerAdapter):
        extra = dict(base.extra)
        logger = base.logger
    else:
        extra = {}
        logger = base
    extra["attempt"] = attempt
    return logging.LoggerAdapter(logger, extra)


def _render_summary_table(rows: List[Dict[str, object]]) -> str:
    headers = ["Connector", "Status", "Attempts", "Rows", "Duration(ms)", "Notes"]
    table_rows = [
        [
            row.get("name", ""),
            row.get("status", ""),
            str(row.get("attempts", "")),
            str(row.get("rows", "")),
            str(row.get("duration_ms", "")),
            row.get("notes", "") or "",
        ]
        for row in rows
    ]
    if not table_rows:
        table_rows = [["-", "-", "0", "0", "0", ""]]
    columns = list(zip(headers, *table_rows))
    widths = [max(len(str(value)) for value in column) for column in columns]
    lines = []
    header_line = " | ".join(header.ljust(width) for header, width in zip(headers, widths))
    lines.append(header_line)
    lines.append("-+-".join("-" * width for width in widths))
    for row in table_rows:
        lines.append(" | ".join(str(value).ljust(width) for value, width in zip(row, widths)))
    return "\n".join(lines)


def _rows_written(before: int, after: int) -> int:
    return max(0, after - before)


def _run_connector(spec: ConnectorSpec, logger: logging.LoggerAdapter) -> Dict[str, object]:
    start = time.perf_counter()
    rows_before = _count_rows(spec.output_path)
    _invoke_connector(spec.path, logger=logger)
    rows_after = _count_rows(spec.output_path)
    duration_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "completed",
        extra={
            "event": "completed",
            "duration_ms": duration_ms,
            "rows_written": _rows_written(rows_before, rows_after),
            "rows_total": rows_after,
        },
    )
    return {"status": "ok", "rows": rows_after, "duration_ms": duration_ms}


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat connector/stub failures as fatal (non-zero exit)",
    )
    parser.add_argument(
        "--mode",
        choices=("real", "stubs", "all"),
        default=None,
        help="override RESOLVER_INGESTION_MODE",
    )
    parser.add_argument(
        "--run-stubs",
        type=int,
        choices=(0, 1),
        default=None,
        help="force running stub connectors (1) or skip them (0)",
    )
    parser.add_argument("--retries", type=int, default=2, help="number of retries per connector")
    parser.add_argument("--retry-base", type=float, default=1.0, help="initial retry delay in seconds")
    parser.add_argument("--retry-max", type=float, default=30.0, help="maximum retry delay in seconds")
    parser.add_argument(
        "--retry-no-jitter",
        action="store_true",
        help="disable jitter for retry backoff",
    )
    parser.add_argument(
        "--connector",
        action="append",
        default=[],
        help="limit run to specific connector(s) (filename without .py or with)",
    )
    parser.add_argument(
        "--log-format",
        choices=("plain", "json"),
        default=None,
        help="console log format (defaults to RUNNER_LOG_FORMAT)",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="log level (defaults to RUNNER_LOG_LEVEL or INFO)",
    )
    if argv is None:
        return parser.parse_args()
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    root_input = sys.argv if argv is None else ['<custom>'] + list(argv)
    requested = { _normalise_name(name) for name in args.connector if name }
    selected: Optional[set[str]] = requested or None

    ingestion_mode = (args.mode or INGESTION_MODE).strip().lower()
    include_stubs = INCLUDE_STUBS if args.run_stubs is None else bool(args.run_stubs)

    log_dir_env = os.environ.get("RUNNER_LOG_DIR")
    if log_dir_env:
        effective_log_dir = Path(log_dir_env).expanduser()
    else:
        effective_log_dir = LOGS_DIR
    os.makedirs(effective_log_dir, exist_ok=True)

    run_id = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    effective_level = args.log_level
    if RESOLVER_DEBUG and not effective_level:
        effective_level = "DEBUG"
    root = init_logger(
        run_id,
        level=effective_level,
        fmt=args.log_format,
        log_dir=effective_log_dir,
    )
    if RESOLVER_DEBUG:
        logging.getLogger().setLevel(logging.DEBUG)
        root.setLevel(logging.DEBUG)
    root.info(
        "initialised logging",
        extra={
            "event": "logging_setup",
            "log_dir": str(effective_log_dir),
            "run_id": run_id,
        },
    )
    log_env_summary(root)
    root.info(
        "parsed arguments",
        extra={
            "event": "args",
            "connector_args": list(args.connector),
            "raw_argv": root_input[1:],
            "mode": args.mode,
            "run_stubs_arg": args.run_stubs,
        },
    )

    warnings.simplefilter("default")

    def _warn_to_log(message, category, filename, lineno, file=None, line=None):  # type: ignore[override]
        root.warning(
            f"{category.__name__}: {message}",
            extra={"event": "warning", "filename": filename, "lineno": lineno},
        )

    warnings.showwarning = _warn_to_log  # type: ignore[assignment]

    if ingestion_mode and ingestion_mode not in {"real", "stubs", "all"}:
        root.error(
            "Unknown RESOLVER_INGESTION_MODE=%s; expected one of real|stubs|all",
            ingestion_mode,
            extra={"event": "config_error"},
        )
        return 0

    real_list = list(REAL)
    stub_list = list(STUBS)
    real_set = set(real_list)
    stub_set = set(stub_list)

    run_real = True
    run_stubs = include_stubs
    if ingestion_mode:
        run_real = ingestion_mode in {"real", "all"}
        run_stubs = ingestion_mode in {"stubs", "all"}
    if FORCE_DTM_STUB:
        run_stubs = True

    if selected is not None:
        unknown = selected - real_set - stub_set
        if unknown:
            for name in sorted(unknown):
                root.warning("Requested connector %s is unknown", name, extra={"event": "unknown_connector"})
            selected -= unknown
        if not selected:
            root.info("No known connectors requested; exiting", extra={"event": "no_connectors"})
            return 0
        run_real = any(name in real_set for name in selected)
        run_stubs = any(name in stub_set for name in selected)

    specs = _build_specs(real_list, stub_list, selected, run_real, run_stubs)
    root.info(
        "planning run",
        extra={
            "event": "plan",
            "requested": sorted(requested),
            "run_real": run_real,
            "run_stubs": run_stubs,
            "count": len(specs),
        },
    )
    if requested:
        specs = [spec for spec in specs if spec.filename in requested]
    if not specs:
        root.info("No connectors to run", extra={"event": "no_connectors"})
        return 0

    connectors_summary: List[Dict[str, object]] = []
    total_start = time.perf_counter()

    for spec in specs:
        child = child_logger(spec.name)
        handler = attach_connector_handler(child, spec.name)
        attempts = 0
        rows = 0
        status = "skipped"
        notes: Optional[str] = None
        connector_start = time.perf_counter()
        try:
            if spec.skip_reason:
                child.warning(
                    "skipped",
                    extra={"event": "skipped", "reason": redact(spec.skip_reason)},
                )
                notes = spec.skip_reason
                status = "skipped"
                attempts = 0
                rows = 0
                continue

            def _attempt(attempt: int) -> Dict[str, object]:
                nonlocal attempts
                attempts = max(attempts, attempt)
                attempt_logger = _with_attempt_logger(child, attempt)
                start_extra = {
                    "event": "start",
                    "attempt": attempt,
                    "kind": spec.kind,
                    "path": str(spec.path),
                }
                for key, value in spec.metadata.items():
                    start_extra[key] = redact(value)
                if spec.summary:
                    start_extra["summary"] = redact(spec.summary)
                attempt_logger.info("starting", extra=start_extra)
                return _run_connector(spec, attempt_logger)

            def _should_retry(exc: BaseException) -> bool:
                exit_code: int | None = None
                stderr_text = ""
                stdout_text = ""
                if isinstance(exc, subprocess.CalledProcessError):
                    exit_code = exc.returncode
                    try:
                        if getattr(exc, "stderr", None):
                            stderr_text = _coerce_process_stream(exc.stderr)
                        if getattr(exc, "stdout", None):
                            stdout_text = _coerce_process_stream(exc.stdout)
                    except Exception:  # noqa: BLE001
                        stderr_text = ""
                        stdout_text = ""
                return _is_retryable_exception(
                    exc,
                    exit_code=exit_code,
                    stderr=stderr_text or None,
                    stdout=stdout_text or None,
                )

            result = retry_call(
                _attempt,
                retries=max(0, args.retries),
                base_delay=args.retry_base,
                max_delay=args.retry_max,
                jitter=not args.retry_no_jitter,
                logger=child,
                connector=spec.name,
                is_retryable=_should_retry,
            )
            rows = int(result.get("rows", 0))
            duration_ms = int((time.perf_counter() - connector_start) * 1000)
            if spec.output_path and spec.output_path.exists() and rows == 0:
                status = "ok-empty"
                notes = "header-only"
            else:
                status = "ok"
            child.info(
                "finished",
                extra={
                    "event": "finished",
                    "status": status,
                    "rows": rows,
                    "attempts": attempts,
                    "duration_ms": duration_ms,
                    "notes": redact(notes) if notes else None,
                },
            )
        except Exception as exc:  # noqa: BLE001
            duration_ms = int((time.perf_counter() - connector_start) * 1000)
            status = "error"
            notes = redact(str(exc))
            rows = 0
            if attempts == 0:
                attempts = 1
            child.error(
                "failed",
                exc_info=exc,
                extra={
                    "event": "error",
                    "rows": rows,
                    "attempts": attempts,
                    "duration_ms": duration_ms,
                },
            )
        finally:
            detach_connector_handler(child, handler)
            connectors_summary.append(
                {
                    "name": spec.name,
                    "status": status,
                    "attempts": attempts,
                    "rows": rows,
                    "duration_ms": int((time.perf_counter() - connector_start) * 1000),
                    "notes": redact(notes) if notes else None,
                    "kind": spec.kind,
                }
            )

    total_duration_ms = int((time.perf_counter() - total_start) * 1000)
    table = _render_summary_table(connectors_summary)
    root.info("Connector summary\n%s", table, extra={"event": "summary_table"})
    root.info(
        "run complete",
        extra={
            "event": "run_summary",
            "run_id": run_id,
            "connectors": [
                {key: value for key, value in entry.items() if key != "kind"}
                for entry in connectors_summary
            ],
            "total_duration_ms": total_duration_ms,
        },
    )

    real_failures = sum(
        1
        for entry in connectors_summary
        if entry.get("kind") == "real" and entry.get("status") == "error"
    )
    stub_failures = sum(
        1
        for entry in connectors_summary
        if entry.get("kind") == "stub" and entry.get("status") == "error"
    )

    if run_real and not run_stubs:
        return 1 if real_failures else 0

    if args.strict and (real_failures or stub_failures):
        return 1
    if FAIL_ON_STUB_ERROR and stub_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
