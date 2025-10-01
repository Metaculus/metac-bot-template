# Ingestion (Scaffold)

This folder holds **connectors** that generate **staging CSVs**. Most are currently stubs
(no external calls yet) so you can exercise the pipeline end-to-end. ReliefWeb is now a
real API client.

**Stubs → Export → Validate → Freeze**

Later (Epic C) we will replace stubs with real API/scraper clients.

## Sources covered

- ReliefWeb — **API connector** (`reliefweb_client.py`) → `staging/reliefweb.csv`
- IFRC GO — stub (`ifrc_go_stub.py`)
- UNHCR ODP — stub (`unhcr_stub.py`)
- IOM DTM — stub (`dtm_stub.py`)
- WHO Emergencies — stub (`who_stub.py`)
- IPC — stub (`ipc_stub.py`)
- EM-DAT — stub (`emdat_stub.py`)
- GDACS — stub (`gdacs_stub.py`)
- Copernicus EMS — stub (`copernicus_stub.py`)
- UNOSAT — stub (`unosat_stub.py`)
- HDX (CKAN) — stub (`hdx_stub.py`)
- ACLED — stub (`acled_stub.py`)
- UCDP — stub (`ucdp_stub.py`)
- FEWS NET — stub (`fews_stub.py`)
- WFP mVAM — stub (`wfp_mvam_stub.py`)
- Gov NDMA — stub (`gov_ndma_stub.py`)

Each connector:
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

Stubs are deterministic samples for now (no network). Replace their internal `make_rows()`
with real pulls in Epic C.

### ReliefWeb specifics

- Window: last `window_days` (configurable in `ingestion/config/reliefweb.yml`)
- Filters: English language; report/appeal/update formats
- Hazards: keyword mapping (conservative)
- Metrics: PIN preferred (if phrase found), else PA; PHE allows `cases` (unit = persons_cases)
- No PDF parsing in v1; title/summary only

Hazards must match resolver/data/shocks.csv (e.g., FL, DR, TC, HW, ACO, ACE, ACC, DI, CU, EC, PHE).

Earthquakes are out of scope by policy and are not included in stubs.

### ReliefWeb troubleshooting

- Set `RESOLVER_DEBUG=1` to log HTTP status, headers, and first 500 chars of any non-200 response.
- Set `RESOLVER_SKIP_RELIEFWEB=1` to bypass the connector (writes an empty header-only CSV), useful if a proxy/WAF blocks the API.

Example:

```bash
$env:RESOLVER_DEBUG="1"
python resolver/ingestion/reliefweb_client.py
```

### ReliefWeb appname

- Default appname: `UNICEF-Resolver-P1L1T6` (set in `ingestion/config/reliefweb.yml`).
- Override without code changes:
  - Windows PowerShell: `$env:RELIEFWEB_APPNAME = "UNICEF-Resolver-P1L1T6"`
  - Bash: `export RELIEFWEB_APPNAME="UNICEF-Resolver-P1L1T6"`

After ReliefWeb approves your appname/IP:
- Remove/avoid `RESOLVER_SKIP_RELIEFWEB=1` in CI/local.
- (Optional) set `RESOLVER_DEBUG=1` once to verify 200/JSON, then turn it off.

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
