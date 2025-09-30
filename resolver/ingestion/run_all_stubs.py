#!/usr/bin/env python3
"""
Run all ingestion stubs to populate resolver/staging/*.csv
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STUBS = [
    "reliefweb_stub.py",
    "ifrc_go_stub.py",
    "unhcr_stub.py",
    "dtm_stub.py",
    "who_stub.py",
    "ipc_stub.py",
]

def main():
    failed = 0
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
