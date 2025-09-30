# Resolver Overview (A1)

## Validation

Run the lightweight validator against any facts CSV/Parquet:

```bash
pip install pandas pyarrow pyyaml
python resolver/tools/validate_facts.py --facts resolver/samples/facts_sample.csv
```

Checks required columns, enums, and dates.

Verifies iso3 exists in resolver/data/countries.csv.

Verifies hazard_code/hazard_label/hazard_class match resolver/data/shocks.csv.

Ensures value >= 0, as_of_date <= publication_date <= today.

Blocks metric = in_need if source_type = media.

Requires unit = persons_cases when metric = cases.

PR checklist addition: ✅ Facts validate against registries (countries.csv, shocks.csv) with validate_facts.py.

## Snapshots

Create a monthly snapshot (validated, parquet + manifest):

```bash
pip install pandas pyarrow pyyaml
python resolver/tools/freeze_snapshot.py --facts resolver/samples/facts_sample.csv --month 2025-09
```

PR checklist addition: ✅ If this PR changes facts or resolver logic, ensure a snapshot plan is documented and (when appropriate) a new snapshot was produced with the freezer.

## Export → Validate → Freeze (quickstart)

```bash
# 1) Export normalized facts from staging files/folder
python resolver/tools/export_facts.py --in resolver/staging --out resolver/exports

# 2) Validate against registries and schema
python resolver/tools/validate_facts.py --facts resolver/exports/facts.csv

# 3) Freeze a monthly snapshot for grading
python resolver/tools/freeze_snapshot.py --facts resolver/exports/facts.csv --month 2025-09


If you see validation errors, fix the staging inputs or tweak resolver/tools/export_config.yml.

## End-to-end (Stubs → Export → Validate → Freeze)

```bash
# 0) Ensure registries exist and include your latest countries/hazards
#    resolver/data/countries.csv
#    resolver/data/shocks.csv

# 1) Generate staging CSVs from stub connectors (no network)
python resolver/ingestion/run_all_stubs.py

# 2) Export canonical facts from staging
python resolver/tools/export_facts.py --in resolver/staging --out resolver/exports

# 3) Validate against registries & schema
python resolver/tools/validate_facts.py --facts resolver/exports/facts.csv

# 4) Freeze a monthly snapshot
python resolver/tools/freeze_snapshot.py --facts resolver/exports/facts.csv --month YYYY-MM
```

This will create:

resolver/staging/*.csv (one per source)

resolver/exports/facts.csv (+ optional Parquet)

resolver/snapshots/YYYY-MM/{facts.parquet,manifest.json}


## Resolve at a cutoff (precedence engine)

Select one authoritative total per `(iso3, hazard_code)` applying A2 policy:

```bash
pip install pandas pyarrow pyyaml python-dateutil
python resolver/tools/precedence_engine.py \
  --facts resolver/exports/facts.csv \
  --cutoff 2025-09-30
```


Outputs:

resolver/exports/resolved.csv

resolver/exports/resolved.jsonl

resolver/exports/resolved_diagnostics.csv (conflict notes)

PR checklist addition: ✅ Precedence config reviewed (tools/precedence_config.yml) and results inspected (exports/resolved*.{csv,jsonl}).


---

**Definition of Done (DoD)**
- `resolver/ingestion/README.md` + `resolver/ingestion/checklist.yml` exist.
- Stubs exist and write staging CSVs: `reliefweb.csv`, `ifrc_go.csv`, `unhcr.csv`, `dtm.csv`, `who.csv`, `ipc.csv`.
- `run_all_stubs.py` runs all stubs and prints **✅ all stubs completed**.
- Root `resolver/README.md` shows the end-to-end commands.
- `resolver/tools/precedence_config.yml` exists with tiers and mapping that match A2.
- `resolver/tools/precedence_engine.py` runs on your exported facts and writes `resolved.csv/jsonl` + diagnostics.
- A local smoke test with `resolver/exports/facts_minimal.csv` succeeds and selects the expected rows.
- `resolver/README.md` updated with usage and a PR checklist line.
