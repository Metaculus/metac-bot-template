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
