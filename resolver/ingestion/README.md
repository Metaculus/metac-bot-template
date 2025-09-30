# Ingestion (Scaffold)

This folder holds **stub connectors** that generate small **staging CSVs** (no external calls yet).
They allow you to test the full pipeline now:

**Stubs → Export → Validate → Freeze**

Later (Epic C) we will replace stubs with real API/scraper clients.

## Sources covered (stubs)
- ReliefWeb (`reliefweb_stub.py`)
- IFRC GO (`ifrc_go_stub.py`)
- UNHCR ODP (`unhcr_stub.py`)
- IOM DTM (`dtm_stub.py`)
- WHO Emergencies (`who_stub.py`)
- IPC (`ipc_stub.py`)

Each stub:
- Reads `resolver/data/countries.csv` and `resolver/data/shocks.csv`
- Produces `resolver/staging/<source>.csv` using the exporter’s expected columns

## Run all stubs

```bash
python resolver/ingestion/run_all_stubs.py


Then:

python resolver/tools/export_facts.py --in resolver/staging --out resolver/exports
python resolver/tools/validate_facts.py --facts resolver/exports/facts.csv
python resolver/tools/freeze_snapshot.py --facts resolver/exports/facts.csv --month YYYY-MM
```

Notes

Stubs are deterministic samples for now (no network). Replace their internal make_rows() with real pulls in Epic C.

Hazards must match resolver/data/shocks.csv (e.g., FL, DR, TC, HW, ACO, ACE, ACC, DI, CU, EC, PHE).

Earthquakes are out of scope by policy and are not included in stubs.
