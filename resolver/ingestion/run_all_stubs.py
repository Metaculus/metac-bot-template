#!/usr/bin/env python3
"""Run ingestion connectors and optional stubs to populate staging CSVs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

INGESTION_MODE = (os.environ.get("RESOLVER_INGESTION_MODE") or "").strip().lower()
INCLUDE_STUBS = os.environ.get("RESOLVER_INCLUDE_STUBS", "0") == "1"
FAIL_ON_STUB_ERROR = os.environ.get("RESOLVER_FAIL_ON_STUB_ERROR", "0") == "1"
FORCE_DTM_STUB = os.environ.get("RESOLVER_FORCE_DTM_STUB", "0") == "1"

REAL = [
    "ifrc_go_client.py",
    "reliefweb_client.py",
    "unhcr_client.py",
    "unhcr_odp_client.py",
    "dtm_client.py",
    "emdat_client.py",
    "gdacs_client.py",
    "hdx_client.py",
    "who_phe_client.py",
    "ipc_client.py",
]

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
    "dtm_client.py": ("RESOLVER_SKIP_DTM", "DTM connector"),
    "emdat_client.py": ("RESOLVER_SKIP_EMDAT", "EM-DAT connector"),
    "gdacs_client.py": ("RESOLVER_SKIP_GDACS", "GDACS connector"),
    "who_phe_client.py": ("RESOLVER_SKIP_WHO", "WHO PHE connector"),
    "ipc_client.py": ("RESOLVER_SKIP_IPC", "IPC connector"),
    "hdx_client.py": ("RESOLVER_SKIP_HDX", "HDX connector"),
}


def _should_skip(script: str) -> bool:
    env_name, label = SKIP_ENVS.get(script, (None, None))
    if env_name and os.environ.get(env_name) == "1":
        print(f"{env_name}=1 — {label} will be skipped")
        return True
    return False


def _run_script(path: Path) -> int:
    print(f"==> running {path.name}")
    proc = subprocess.run([sys.executable, str(path)])
    return proc.returncode


def main() -> None:
    if INGESTION_MODE and INGESTION_MODE not in {"real", "stubs", "all"}:
        print(
            f"Unknown RESOLVER_INGESTION_MODE={INGESTION_MODE!r}; expected one of real|stubs|all"
        )
        return

    if INGESTION_MODE:
        run_real = INGESTION_MODE in {"real", "all"}
        run_stubs = INGESTION_MODE in {"stubs", "all"}
    else:
        run_real = True
        run_stubs = INCLUDE_STUBS

    if FORCE_DTM_STUB:
        run_stubs = True

    if run_real:
        real_failed = 0
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
                real_failed += 1

        if real_failed:
            print(f"{real_failed} real connector(s) failed", file=sys.stderr)
            sys.exit(1)

    if not run_stubs:
        return

    stub_failed = 0
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
            stub_failed += 1

    if stub_failed:
        print(f"{stub_failed} stub(s) failed", file=sys.stderr)
        if FAIL_ON_STUB_ERROR:
            sys.exit(1)
    else:
        print("All stubs ran successfully")


if __name__ == "__main__":
    main()
