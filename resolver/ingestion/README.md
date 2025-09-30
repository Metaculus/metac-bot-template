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
- EM-DAT (`emdat_stub.py`)
- GDACS (`gdacs_stub.py`)
- Copernicus EMS (`copernicus_stub.py`)
- UNOSAT (`unosat_stub.py`)
- HDX (CKAN) (`hdx_stub.py`)
- ACLED (`acled_stub.py`)
- UCDP (`ucdp_stub.py`)
- FEWS NET (`fews_stub.py`)
- WFP mVAM (`wfp_mvam_stub.py`)
- Gov NDMA (`gov_ndma_stub.py`)

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

## Source notes (what each adds)

- **EM-DAT** — standardized disaster records and “people affected”; lagged but consistent.
- **GDACS** — near-real-time alerts and modeled impact for hydro-meteo hazards.
- **Copernicus EMS / UNOSAT** — activation footprints, damage/exposure mapping (good PA proxies).
- **HDX (CKAN)** — dataset discovery hub; many country datasets flow here.
- **ACLED / UCDP** — conflict event data (drivers/attribution context; not PIN).
- **FEWS NET** — early warning analyses; aligns with IPC phases for DR/EC.
- **WFP mVAM** — market/price/food security indicators (context for EC/DR).
- **Gov NDMA** — national sitreps; often earliest official “people affected”.

> **Reminder:** Stubs emit plausible demo rows only. Final **resolution** still follows A2 precedence (PIN preferred; PA proxy), and Tier-2/3 sources should not override Tier-1.
