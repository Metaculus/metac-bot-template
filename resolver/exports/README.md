# Exports

This folder receives normalized resolver outputs produced by the exporter:

- `facts.csv` — canonical CSV matching the Data Dictionary
- `facts.parquet` — same rows in columnar format (optional)

## How to create
```bash
pip install pandas pyarrow pyyaml
python resolver/tools/export_facts.py --in resolver/staging --out resolver/exports


Then:

python resolver/tools/validate_facts.py --facts resolver/exports/facts.csv
python resolver/tools/freeze_snapshot.py --facts resolver/exports/facts.csv --month YYYY-MM


---

### 👉 Update: `resolver/README.md` (append this quickstart)

```md
## Export → Validate → Freeze (quickstart)

```bash
# 1) Export normalized facts from staging files/folder
python resolver/tools/export_facts.py --in resolver/staging --out resolver/exports

# 2) Validate against registries and schema
python resolver/tools/validate_facts.py --facts resolver/exports/facts.csv

# 3) Freeze a monthly snapshot for grading
python resolver/tools/freeze_snapshot.py --facts resolver/exports/facts.csv --month 2025-09


If you see validation errors, fix the staging inputs or tweak resolver/tools/export_config.yml.


---

**Definition of Done (DoD)**  
- `resolver/tools/export_facts.py` and `resolver/tools/export_config.yml` exist with the exact content above.  
- `resolver/staging/sample_source.csv` and `resolver/exports/README.md` added.  
- `resolver/README.md` updated with the quickstart.  
- Local smoke test (on your machine):
