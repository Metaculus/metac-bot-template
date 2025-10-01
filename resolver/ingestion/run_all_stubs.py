#!/usr/bin/env python3
"""
Run all ingestion stubs to populate resolver/staging/*.csv
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STUBS = [
    "ifrc_go_stub.py",
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
    reliefweb_client = ROOT / "reliefweb_client.py"
    if reliefweb_client.exists():
        print("==> running reliefweb_client.py (real API)")
        res = subprocess.run([sys.executable, str(reliefweb_client)])
        if res.returncode != 0:
            print(
                "ReliefWeb client failed; continuing with other sources…",
                file=sys.stderr,
            )
    else:
        print("reliefweb_client.py missing; skipping real connector", file=sys.stderr)

    for stub in STUBS:
        path = ROOT / stub
        print(f"==> running {stub}")
        res = subprocess.run([sys.executable, str(path)])
        if res.returncode != 0:
            failed += 1
    if failed:
        print(f"{failed} stub(s) failed", file=sys.stderr)
        sys.exit(1)
    print("✅ all stubs completed")

if __name__ == "__main__":
    main()
