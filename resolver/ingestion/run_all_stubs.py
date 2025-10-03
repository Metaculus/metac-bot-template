#!/usr/bin/env python3
"""Run ingestion connectors and optional stubs to populate staging CSVs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import yaml

ROOT = Path(__file__).resolve().parent
STAGING = ROOT.parent / "staging"
CONFIG_DIR = ROOT / "config"

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


def _load_yaml(path: Optional[Path]) -> dict:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _count_rows(path: Path) -> int:
    if not path.exists():
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


def _should_skip(script: str) -> bool:
    env_name, label = SKIP_ENVS.get(script, (None, None))
    if env_name and os.environ.get(env_name) == "1":
        print(f"{env_name}=1 — {label} will be skipped")
        return True
    return False


def _run_script(path: Path) -> int:
    print(f"==> running {path.name}")
    try:
        proc = subprocess.run([sys.executable, str(path)])
        return proc.returncode
    except OSError as exc:
        print(f"{path.name} failed to start: {exc}", file=sys.stderr)
        return 1


def _safe_summary(name: str) -> None:
    try:
        summary_line = _summarise_connector(name)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[summary] {name} failed: {exc}", file=sys.stderr)
        return
    if summary_line:
        print(summary_line)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat connector/stub failures as fatal (non-zero exit)",
    )
    return parser.parse_args(argv or [])


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    strict_mode = bool(args.strict)

    if INGESTION_MODE and INGESTION_MODE not in {"real", "stubs", "all"}:
        print(
            f"Unknown RESOLVER_INGESTION_MODE={INGESTION_MODE!r}; expected one of real|stubs|all"
        )
        return 0

    if INGESTION_MODE:
        run_real = INGESTION_MODE in {"real", "all"}
        run_stubs = INGESTION_MODE in {"stubs", "all"}
    else:
        run_real = True
        run_stubs = INCLUDE_STUBS

    if FORCE_DTM_STUB:
        run_stubs = True

    real_failures = 0
    if run_real:
        for name in REAL:
            path = ROOT / name
            if _should_skip(name):
                continue
            if not path.exists():
                print(f"{name} missing; skipping", file=sys.stderr)
                continue
            rc = _run_script(path)
            if rc != 0:
                print(f"{name} failed with rc={rc}", file=sys.stderr)
                real_failures += 1

        for summary_name in SUMMARY_TARGETS:
            if summary_name in REAL:
                _safe_summary(summary_name)

    if not run_stubs:
        return 1 if (strict_mode and real_failures) else 0

    stub_failures = 0
    for name in STUBS:
        if FORCE_DTM_STUB and not INCLUDE_STUBS and name != "dtm_stub.py":
            continue
        path = ROOT / name
        if not path.exists():
            print(f"{name} missing; skipping", file=sys.stderr)
            continue
        rc = _run_script(path)
        if rc != 0:
            print(f"{name} failed with rc={rc}", file=sys.stderr)
            stub_failures += 1

    if stub_failures:
        print(f"{stub_failures} stub(s) failed", file=sys.stderr)
    else:
        print("All stubs ran successfully")

    strict_failures = 0
    if strict_mode and (real_failures or stub_failures):
        strict_failures = real_failures + stub_failures
    elif FAIL_ON_STUB_ERROR and stub_failures:
        strict_failures = stub_failures

    if strict_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
