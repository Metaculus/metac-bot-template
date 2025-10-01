# Resolver — Data Contract Tests

These tests enforce basic contracts across:
- Registries (`resolver/data/*.csv`)
- Exports (`resolver/exports/facts.csv`)
- Resolved outputs (`resolver/exports/resolved*.{csv,jsonl}`)
- Review queue (`resolver/review/review_queue.csv`)
- Snapshots (`resolver/snapshots/YYYY-MM/facts.parquet`), if present
- Remote-first state files under `resolver/state/**/exports/*.csv`

## Run locally (cross-platform)

First install dev requirements:

```powershell
# Windows PowerShell
python -m pip install -r resolver/requirements-dev.txt
```

```bash
# macOS/Linux
python3 -m pip install -r resolver/requirements-dev.txt
```

Then run tests:

```bash
python -m pytest resolver/tests -q
```

Tests will skip gracefully if an expected file isn't present (e.g., snapshots),
but will fail if a file exists and violates the contract.

### Hermetic connector tests
Header tests set `RESOLVER_SKIP_IFRCGO=1` and `RESOLVER_SKIP_RELIEFWEB=1` so no network is required.
Each connector must still produce a CSV with the canonical header (even if empty).
