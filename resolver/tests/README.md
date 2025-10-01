# Resolver — Data Contract Tests

These tests enforce basic contracts across:
- Registries (`resolver/data/*.csv`)
- Exports (`resolver/exports/facts.csv`)
- Resolved outputs (`resolver/exports/resolved*.{csv,jsonl}`)
- Review queue (`resolver/review/review_queue.csv`)
- Snapshots (`resolver/snapshots/YYYY-MM/facts.parquet`), if present
- Remote-first state files under `resolver/state/**/exports/*.csv`

## Run locally (cross-platform)

```bash
python -m pytest resolver/tests -q
```

Use python -m pytest on Windows to avoid PATH issues (fixes “pytest is not recognized”).

**CI already uses `pytest` in PATH** (via `pip install pytest`). That’s fine for Linux runners; no change needed there.

Tests will skip gracefully if an expected file isn't present (e.g., snapshots),
but will fail if a file exists and violates the contract.

### Hermetic connector tests
Header tests set `RESOLVER_SKIP_IFRCGO=1` and `RESOLVER_SKIP_RELIEFWEB=1` so no network is required.
Each connector must still produce a CSV with the canonical header (even if empty).
