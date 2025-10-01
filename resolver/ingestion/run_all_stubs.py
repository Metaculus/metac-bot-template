#!/usr/bin/env python3
"""Run all ingestion stubs to populate resolver/staging/*.csv."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STUBS = [
    "ifrc_go_client.py",      # real connector (fail-soft on error/skip)
    "reliefweb_client.py",    # real connector (may be skipped via env)
    "unhcr_stub.py",
    "dtm_stub.py",
    "who_stub.py",
    "ipc_stub.py",
    "emdat_stub.py",
    "gdacs_stub.py",
    "copernicus_stub.py",
    "unosat_stub.py",
    "hdx_stub.py",
    "acled_stub.py",
    "ucdp_stub.py",
    "fews_stub.py",
    "wfp_mvam_stub.py",
    "gov_ndma_stub.py",
]

def main():
    failed = 0
    env = os.environ.copy()

    for script in STUBS:
        path = ROOT / script

        if script == "ifrc_go_client.py":
            if env.get("RESOLVER_SKIP_IFRCGO") == "1":
                print("RESOLVER_SKIP_IFRCGO=1 — IFRC GO connector will be skipped")
                continue
            if not path.exists():
                print("ifrc_go_client.py missing; skipping real connector", file=sys.stderr)
                continue
            print("==> running ifrc_go_client.py (real connector)")
            try:
                res = subprocess.run([sys.executable, str(path)], env=env)
            except Exception as exc:
                print(f"IFRC GO client raised {exc}; continuing with other sources…", file=sys.stderr)
                continue
            if res.returncode != 0:
                print("IFRC GO client failed; continuing with other sources…", file=sys.stderr)
            continue

        if script == "reliefweb_client.py":
            if env.get("RESOLVER_SKIP_RELIEFWEB") == "1":
                print("RESOLVER_SKIP_RELIEFWEB=1 — ReliefWeb connector will be skipped")
                continue
            if not path.exists():
                print("reliefweb_client.py missing; skipping real connector", file=sys.stderr)
                continue
            print("==> running reliefweb_client.py (real connector)")
            res = subprocess.run([sys.executable, str(path)], env=env)
            if res.returncode != 0:
                print("ReliefWeb client failed; continuing with other sources…", file=sys.stderr)
            continue

        print(f"==> running {script}")
        res = subprocess.run([sys.executable, str(path)])
        if res.returncode != 0:
            failed += 1

    if failed:
        print(f"{failed} stub(s) failed", file=sys.stderr)
        sys.exit(1)
    print("✅ all stubs completed")

if __name__ == "__main__":
    main()
