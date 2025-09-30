# Resolver API (FastAPI)

Serve the resolver over HTTP.

## Run locally

```bash
pip install -r resolver/requirements.txt
uvicorn resolver.api.app:app --reload --port 8000
```

Open the docs:

http://127.0.0.1:8000/docs

http://127.0.0.1:8000/health

## Examples

### By codes

```bash
curl "http://127.0.0.1:8000/resolve?iso3=PHL&hazard_code=TC&cutoff=2025-09-30"
```

### By names

```bash
curl "http://127.0.0.1:8000/resolve?country=Philippines&hazard=Tropical%20Cyclone&cutoff=2025-09-30"
```

## Selection logic

- Past months → reads `snapshots/YYYY-MM/facts.parquet`
- Current month → prefers `exports/resolved_reviewed.csv`, else `exports/resolved.csv`

Exactly like the CLI.

## Notes

- Keep `resolver/data/countries.csv` and `resolver/data/shocks.csv` current.
- If a snapshot for the target month does not exist, the API falls back to exports/.
