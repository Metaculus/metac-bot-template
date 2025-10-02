#!/usr/bin/env python3
"""Run all ingestion stubs to populate resolver/staging/*.csv."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STUBS = [
    "ifrc_go_client.py",      # real connector (fail-soft on error/skip)
    "unhcr_client.py",        # real connector (fail-soft/skip-capable)
    "unhcr_odp_client.py",    # real connector (fail-soft/skip-capable)
    "dtm_client.py",          # real connector (fail-soft/skip-capable)
    "who_phe_client.py",      # real connector (fail-soft/skip-capable)
    "ipc_client.py",          # real connector (fail-soft/skip-capable)
    "reliefweb_client.py",    # real connector (may be skipped via env)
    "hdx_client.py",          # real connector (fail-soft/skip-capable)
    "dtm_stub.py",
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
                failed += 1
            continue

        if script == "unhcr_client.py":
            if env.get("RESOLVER_SKIP_UNHCR") == "1":
                print("RESOLVER_SKIP_UNHCR=1 — UNHCR connector will be skipped")
                continue
            if not path.exists():
                print("unhcr_client.py missing; skipping real connector", file=sys.stderr)
                continue
            print("==> running unhcr_client.py (real connector)")
            try:
                res = subprocess.run([sys.executable, str(path)], env=env)
            except Exception as exc:
                print(f"UNHCR client raised {exc}; continuing with other sources…", file=sys.stderr)
                continue
            if res.returncode != 0:
                print("UNHCR client failed; continuing with other sources…", file=sys.stderr)
            continue

        if script == "unhcr_odp_client.py":
            if env.get("RESOLVER_SKIP_UNHCR_ODP") == "1":
                print("RESOLVER_SKIP_UNHCR_ODP=1 — UNHCR ODP connector will be skipped")
                continue
            if not path.exists():
                print("unhcr_odp_client.py missing; skipping real connector", file=sys.stderr)
                continue
            print("==> running unhcr_odp_client.py (real ODP)")
            try:
                res = subprocess.run([sys.executable, str(path)], env=env)
            except Exception as exc:
                print(f"UNHCR ODP client raised {exc}; continuing with other sources…", file=sys.stderr)
                continue
            if res.returncode != 0:
                print("UNHCR ODP client failed; continuing with other sources…", file=sys.stderr)
                failed += 1
            continue

        if script == "dtm_client.py":
            if env.get("RESOLVER_SKIP_DTM") == "1":
                print("RESOLVER_SKIP_DTM=1 — DTM connector will be skipped")
                continue
            if not path.exists():
                print("dtm_client.py missing; skipping real connector", file=sys.stderr)
                continue
            print("==> running dtm_client.py (real connector)")
            try:
                mod = importlib.import_module("resolver.ingestion.dtm_client")
                ok = mod.main()
            except Exception as exc:
                print(f"DTM client raised {exc}; continuing with other sources…", file=sys.stderr)
                failed += 1
                continue
            if not ok:
                print("DTM client produced no rows", file=sys.stderr)
            continue

        if script == "who_phe_client.py":
            if env.get("RESOLVER_SKIP_WHO") == "1":
                print("RESOLVER_SKIP_WHO=1 — WHO PHE connector will be skipped")
                continue
            if not path.exists():
                print("who_phe_client.py missing; skipping WHO connector", file=sys.stderr)
                continue
            print("==> running who_phe_client.py")
            try:
                mod = importlib.import_module("resolver.ingestion.who_phe_client")
                ok = mod.main()
            except Exception as exc:
                print(f"WHO PHE client raised {exc}; continuing with other sources…", file=sys.stderr)
                failed += 1
                continue
            if not ok:
                print("WHO PHE connector produced no rows", file=sys.stderr)
            continue

        if script == "ipc_client.py":
            if env.get("RESOLVER_SKIP_IPC") == "1":
                print("RESOLVER_SKIP_IPC=1 — IPC connector will be skipped")
                continue
            if not path.exists():
                print("ipc_client.py missing; skipping IPC connector", file=sys.stderr)
                continue
            print("==> running ipc_client.py")
            try:
                mod = importlib.import_module("resolver.ingestion.ipc_client")
                ok = mod.main()
            except Exception as exc:
                print(f"IPC connector raised {exc}; continuing with other sources…", file=sys.stderr)
                failed += 1
                continue
            if not ok:
                print("IPC connector produced no rows", file=sys.stderr)
            continue

        if script == "hdx_client.py":
            if env.get("RESOLVER_SKIP_HDX") == "1":
                print("RESOLVER_SKIP_HDX=1 — HDX connector will be skipped")
                continue
            if not path.exists():
                print("hdx_client.py missing; skipping real connector", file=sys.stderr)
                continue
            print("==> running hdx_client.py (real connector)")
            try:
                mod = importlib.import_module("resolver.ingestion.hdx_client")
                mod.main()
            except Exception as exc:
                print(f"HDX client raised {exc}; continuing with other sources…", file=sys.stderr)
                failed += 1
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
