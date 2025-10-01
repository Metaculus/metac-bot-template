# Ingestion (Scaffold)

This folder holds **connectors** that generate **staging CSVs**. Most are currently stubs
(no external calls yet) so you can exercise the pipeline end-to-end. ReliefWeb and IFRC
GO are now real API clients.

**Stubs → Export → Validate → Freeze**

Later (Epic C) we will replace stubs with real API/scraper clients.

## Sources covered

- ReliefWeb — **API connector** (`reliefweb_client.py`) → `staging/reliefweb.csv`
- IFRC GO — **API connector** (`ifrc_go_client.py`) → `staging/ifrc_go.csv`
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

## IFRC GO (Admin v2) — real connector

- **Endpoints:** Admin v2 (`https://goadmin.ifrc.org/api/v2/`) — see GO wiki and Swagger for details.  
  - Field Reports: `field-report/`  (data dictionary on GO wiki)  
  - Appeals: `appeal/`              (data dictionary on GO wiki)  
  - Situation Reports: `situation_report/` (document stream)  
  Docs & references: GO API overview + data dictionaries + Swagger UI. :contentReference[oaicite:2]{index=2}

- **Auth:** Public for many endpoints. Optional `GO_API_TOKEN` header supported if required later.

- **Selection:** We prefer numeric fields like `num_affected`/`people_in_need` when present; else conservative text heuristics (title/summary) with regex.  
  `metric_preference = in_need → affected → cases` (PHE only for `cases` with unit `persons_cases`).

- **Env switches:**
  - `RESOLVER_SKIP_IFRCGO=1` — skip the connector (writes header-only CSV)
  - `RESOLVER_DEBUG=1` — verbose logging

**Windowing & late edits**
The connector now filters at the API with:
- `created_at__gte=<YYYY-MM-DD>`
- `updated_at__gte=<YYYY-MM-DD>`
and early-exits paging when consecutive pages are fully older than the window. Hard caps guard against runaway pagination:
`MAX_PAGES=50`, `MAX_RESULTS=5000`, and debug is throttled with `DEBUG_EVERY=10`.

**Admin v2 details expansion**

GO Admin v2 returns related fields as IDs unless you request `*_details`.
The connector now requests `countries_details` and `disaster_type_details` via the `fields` param,
and handles both shapes. This fixes cases where `countries` was `[123, 456]` (IDs) and avoids crashes.

Tests include a **connector header check** that ensures each connector always writes a canonical CSV
(even when the API is unavailable), keeping the pipeline green and contracts stable.

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
### ReliefWeb WAF / appname

- The API requires `appname` (v2). From **1 Nov 2025** you need a **pre-approved appname**. See docs: <https://apidoc.reliefweb.int/parameters>
- If requests return `202` with header `x-amzn-waf-action: challenge`, AWS WAF is blocking non-browser clients. The connector will **fail-soft** (empty `staging/reliefweb.csv`) and log a message.
- Workarounds:
  1. Register a pre-approved appname with ReliefWeb and request allowlisting for your appname/IP.
  2. Temporarily set `RESOLVER_SKIP_RELIEFWEB=1` in CI/local to bypass the connector.
  3. Use `RESOLVER_DEBUG=1` to print HTTP status/headers.

Until allowlisted, configure CI with:

```yaml
env:
  RESOLVER_SKIP_RELIEFWEB: "1"
```

Flip it to `"0"` once ReliefWeb confirms your appname/IP.

## UNHCR Population — real connector

- **Endpoint (configurable):** `https://api.unhcr.org/population/v1/` (see `config/unhcr.yml`).
- **What we extract:** recent **asylum applications** per country of asylum → mapped to **Displacement Influx (DI)** with `metric=affected`, `unit=persons`.
- **Dates:** Prefer year+month (mid-month as_of), else `updated_at`; publication_date mirrors as_of unless `record_date` present.
- **Env:** `RESOLVER_SKIP_UNHCR=1`, `RESOLVER_DEBUG=1`, `RESOLVER_MAX_PAGES`, `RESOLVER_MAX_RESULTS`, `RESOLVER_DEBUG_EVERY`.
- **Fail-soft:** Writes header-only CSV on errors so the pipeline keeps running.

**Endpoint & mapping**

UNHCR’s public API exposes `/asylum-applications/`, `/population/`, `/asylum-decisions/` (no `/arrivals/`). We query
`/asylum-applications/` with `cf_type=ISO` and a **year window** (this year + previous if needed) and map the count to
**DI (Displacement Influx)** with `metric=affected, unit=persons`. See the official API docs.  

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
