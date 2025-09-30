# Monthly Snapshots

Snapshots are **immutable, monthly freezes** of the facts table used for grading “By <DATE>” questions.

## When
- Freeze at **23:59 Europe/Istanbul** on the **last calendar day**.
- You can also produce ad-hoc snapshots for testing.

## What gets written
- `snapshots/YYYY-MM/facts.parquet` — row-for-row copy of the facts at freeze time
- `snapshots/YYYY-MM/manifest.json` — metadata:
  - `created_at_utc`
  - `source_file` (if produced from a CSV/Parquet export)
  - `source_commit_sha` (if available, e.g., in CI)
  - `rows`

## How to create one locally

```bash
pip install pandas pyarrow pyyaml
python resolver/tools/freeze_snapshot.py --facts resolver/samples/facts_sample.csv --month 2025-09


This will:

Validate the facts using resolver/tools/validate_facts.py

Write snapshots/2025-09/facts.parquet

Write snapshots/2025-09/manifest.json

Overwriting

Snapshots are intended to be immutable. The freezer blocks overwrites by default.
If you must regenerate (e.g., test fixture), pass --overwrite.

Grading rule

Grading uses the snapshot matching the question month (e.g., “By 2025-09-30” → snapshots/2025-09/).

Live dashboards may read from the current facts table.


---
