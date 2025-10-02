# Resolver CLI

Answer questions like:
**“By `<DATE>`, how many people `<METRIC>` due to `<HAZARD>` in `<COUNTRY>`?”**

## Usage

```bash
# name/label inputs
python resolver/cli/resolver_cli.py \
  --country "Philippines" \
  --hazard "Tropical Cyclone" \
  --cutoff 2025-09-30

# code inputs
python resolver/cli/resolver_cli.py --iso3 PHL --hazard_code TC --cutoff 2025-09-30

# request stock totals instead of monthly new deltas
python resolver/cli/resolver_cli.py --iso3 PHL --hazard_code TC --cutoff 2025-09-30 --series stock

# JSON-only output for automation
python resolver/cli/resolver_cli.py --iso3 ETH --hazard_code DR --cutoff 2025-08-31 --json_only
```

### Data selection rules

- **Past months** → uses `snapshots/YYYY-MM/facts.parquet` (preferred)
- **Current month** → prefers `exports/resolved_reviewed.csv`, else `exports/resolved.csv`
- **Series selection** → defaults to monthly `new` deltas; use `--series stock` to return totals. Missing deltas emit a note and fall back to stock data.
- Returns one record per `(iso3, hazard_code)` at the cutoff (PIN preferred, else PA) following upstream policy.

### Dependencies

```bash
pip install pandas pyarrow
```

### Notes

- Keep `resolver/data/countries.csv` and `resolver/data/shocks.csv` up to date.
- If you have not frozen the month yet, the CLI will use current `exports/` outputs.
