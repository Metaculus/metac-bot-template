# Resolver — Data Contract Tests

These tests enforce basic contracts across:
- Registries (`resolver/data/*.csv`)
- Exports (`resolver/exports/facts.csv`)
- Resolved outputs (`resolver/exports/resolved*.{csv,jsonl}`)
- Review queue (`resolver/review/review_queue.csv`)
- Snapshots (`resolver/snapshots/YYYY-MM/facts.parquet`), if present
- Remote-first state files under `resolver/state/**/exports/*.csv`

## Run locally

```bash
pip install pytest pandas pyarrow pyyaml python-dateutil
pytest resolver/tests -q
```

Tests will skip gracefully if an expected file isn't present (e.g., snapshots),
but will fail if a file exists and violates the contract.
