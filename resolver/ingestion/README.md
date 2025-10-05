# Ingestion (Scaffold)

This folder holds **connectors** that generate **staging CSVs**. Most are currently stubs
(no external calls yet) so you can exercise the pipeline end-to-end. ReliefWeb and IFRC
GO are now real API clients.

**Stubs → Export → Validate → Freeze**

Later (Epic C) we will replace stubs with real API/scraper clients.

## Sources covered

- ReliefWeb — **API connector** (`reliefweb_client.py`) → `staging/reliefweb.csv`
- IFRC GO — **API connector** (`ifrc_go_client.py`) → `staging/ifrc_go.csv`
- UNHCR ODP — **API connector** (`unhcr_odp_client.py`) → `staging/unhcr_odp.csv`
- IOM DTM — **API connector** (`dtm_client.py`) → `staging/dtm_displacement.csv`
- WHO Public Health Emergencies — **API connector** (`who_phe_client.py`) → `staging/who_phe.csv`
- WHO Emergencies — stub (`who_stub.py`)
- IPC — stub (`ipc_stub.py`)
- EM-DAT — **file connector** (`emdat_client.py`) → `staging/emdat_pa.csv`
- GDACS — **API connector** (`gdacs_client.py`) → `staging/gdacs_signals.csv`
- Copernicus EMS — stub (`copernicus_stub.py`)
- UNOSAT — stub (`unosat_stub.py`)
- HDX (CKAN) — **API connector** (`hdx_client.py`) → `staging/hdx.csv`
- ACLED — **API connector** (`acled_client.py`) → `staging/acled.csv`
- UCDP — stub (`ucdp_stub.py`)
- FEWS NET — stub (`fews_stub.py`)
- WorldPop — **denominator loader** (`worldpop_client.py`) → `staging/worldpop_denominators.csv`
- WFP mVAM — **API connector** (`wfp_mvam_client.py`) → `staging/wfp_mvam.csv`
- Gov NDMA — stub (`gov_ndma_stub.py`)

Each connector:
- Reads `resolver/data/countries.csv` and `resolver/data/shocks.csv`
- Produces `resolver/staging/<source>.csv` using the exporter’s expected columns

## Connector toggles & outputs

| Connector | Enabled via | Primary source | Monthly allocation | Dedup strategy | Output |
| --- | --- | --- | --- | --- | --- |
| GDACS alerts | `ingestion/config/gdacs.yml` → `enabled` | GDACS public API (`geteventlist`) | Alert start month bucket | `(iso3, hazard, month_start, raw_event_id)` keep latest `as_of` | `resolver/staging/gdacs_signals.csv` |
| EM-DAT impacts | `ingestion/config/emdat.yml` → `enabled` | Licensed EM-DAT CSV/HDX mirror | Linear days-in-month split for spans >14 days | `(iso3, hazard, month_start, raw_event_id, value_type)` | `resolver/staging/emdat_pa.csv` |
| IOM DTM displacement | `ingestion/config/dtm.yml` → `enabled` | File/HDX stock tables | Stock→flow (`diff_nonneg`) with optional admin totals | `(iso3, admin1, month_start, source_id)` | `resolver/staging/dtm_displacement.csv` |
| WorldPop denominators | `ingestion/config/worldpop.yml` → `enabled` | Cached WorldPop national totals | Annual totals (no allocation) | `(iso3, year)` upsert on rerun | `resolver/staging/worldpop_denominators.csv` |

### GDACS severity & EM-DAT allocation rules

- **GDACS alerts**: Green → `0`, Orange → `1`, Red → `2`. Alerts are bucketed to the month of the
  first alert date. If the same `(iso3, hazard, month, raw_event_id)` appears again the connector
  keeps the latest `as_of` timestamp.
- **EM-DAT events**: Events that span more than 14 days are split across months using
  days-in-month weights (last month keeps the remainder after rounding). Shorter incidents keep the
  full total on the start month. Deduplication keeps the newest record per `(iso3, hazard,
  month_start, raw_event_id, value_type)`.
- **DTM displacement**: Stock tables convert to monthly flows using the non-negative
  difference rule; flows are preserved as-is. National totals equal the sum of admin1 rows when the
  config sets `admin_agg: both`.
  - Deduplication keeps the most recent `as_of` per `(country, admin1, month, source)`; older or equal timestamps are skipped to prevent regressions.
- **WorldPop denominators**: Upserts replace previously written `(iso3, year)` rows so reruns update
  `as_of` timestamps without duplicating records.

## Run all stubs

```bash
python resolver/ingestion/run_all_stubs.py


Then:

python resolver/tools/export_facts.py --in resolver/staging --out resolver/exports
python resolver/tools/validate_facts.py --facts resolver/exports/facts.csv
python resolver/tools/freeze_snapshot.py --facts resolver/exports/facts.csv --month YYYY-MM
```

### Structured logging & retries

`run_all_stubs.py` now emits detailed run metadata and per-connector logs. Each run creates
plain text and JSONL logs in `resolver/logs/ingestion/` (configurable via `RUNNER_LOG_DIR`).
A subdirectory per run (named with the UTC start timestamp) contains per-connector logs, so
`resolver/logs/ingestion/20250101-010203/ifrc_go_stub.log` holds only the IFRC stub output
for that run.

Key CLI switches:

- `--connector foo --connector bar` — only run the selected connectors/stubs.
- `--retries N` — retry flaky connectors with exponential backoff (defaults to `2`).
- `--retry-base`, `--retry-max`, `--retry-no-jitter` — tune the retry backoff curve.
- `--strict` — exit non-zero if any connector fails (default is soft-fail to keep CI green).
- `--log-format plain|json`, `--log-level INFO|DEBUG|...` — tweak console verbosity.

The logger also prints an environment summary (git commit, Python version, selected env
flags) to help debug CI runs. Sensitive values (tokens, secrets, long random strings) are
redacted automatically in both console and file output.

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

Hazards must match resolver/data/shocks.csv (e.g., FL, DR, TC, HW, ACO, ACE, DI, CU, EC, PHE).

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
- **Env:** `RESOLVER_SKIP_UNHCR=1`, `RESOLVER_DEBUG=1`, `RESOLVER_MAX_RESULTS`, `RESOLVER_DEBUG_EVERY`.
- **Fail-soft:** Writes header-only CSV on errors so the pipeline keeps running.

*Monthly handling*: The Refugee Statistics API is documented as **yearly**. If the response includes a `month`/`date` field or
if `params.granularity: month` is used and supported, the connector derives a monthly `as_of` (YYYY-MM-15) and includes it in
`event_id` so different months never deduplicate. Otherwise, it falls back to annual `as_of` on 31 December. :contentReference[oaicite:2]{index=2}

**Endpoint & mapping**

UNHCR’s public API exposes `/asylum-applications/`, `/population/`, `/asylum-decisions/` (no `/arrivals/`). We query
`/asylum-applications/` with `cf_type=ISO` and a **year window** (this year + previous if needed) and map the count to
**DI (Displacement Influx)** with `metric=affected, unit=persons`. See the official API docs.

## IOM DTM — real connector

- **Phase 1 source:** HDX mirrors of DTM datasets (CSV/XLSX). Discovery terms configurable via `ingestion/config/dtm.yml`.
- **Scope:** Monthly-first ingestion of displacement / needs (`metric ∈ {in_need, affected}`) by ISO3 and mapped shock.
- **Series semantics:** Incident / flow series pass through. Cumulative series are converted to **month-over-month deltas**
  (first month dropped unless `DTM_ALLOW_FIRST_MONTH=1`). Negative deltas clip to `0`.
- **Aggregation:** Subnational rows are summed to national totals before the delta step. Output rows are always monthly and
  carry deterministic `event_id`s derived from `(iso3, hazard_code, metric, as_of_month, value, source_url)`.
- **Deduplication:** `as_of` aware. Prefer per-row timestamps (e.g. `updated_at`, `report_date`), then a date embedded in the
  filename, then file modified time, and only lastly the run date. The newest `as_of` wins per `(country, admin1, month,
  source)` key.
- **Hazard mapping:** Uses the `shock_keywords` lexicon from `config/dtm.yml`. Ambiguous titles fall back to
  `hazard_code=multi`; pure movement defaults to `displacement_influx` unless overridden by `DTM_DEFAULT_HAZARD`.
- **Publisher metadata:** `publisher="IOM-DTM"`, `source_type="cluster"`, `method="DTM; HXL-aware; monthly-first; delta-on-cumulative"`.

**Env toggles:**

- `RESOLVER_SKIP_DTM=1` — skip the connector (still writes an empty `staging/dtm_displacement.csv` with canonical headers).
- `RESOLVER_MAX_RESULTS=<int>` — optional cap for emitted rows.
- `RESOLVER_DEBUG=1` — verbose logging (shared convention across ingestion clients).
- `DTM_ALLOW_FIRST_MONTH=1` — include the first month from cumulative series as a delta (default 0 → drop it).
- `DTM_DEFAULT_HAZARD=<key>` — override the fallback hazard keyword (default `displacement_influx`).
- `DTM_BASE=<url>` — override the base discovery endpoint (defaults to `https://data.humdata.org`).
- `RELIEFWEB_APPNAME` — reused for User-Agent hints when hitting HDX mirrors.

**Compatibility helpers:** The module continues to expose the historic
`SERIES_INCIDENT`/`SERIES_CUMULATIVE` constants, `load_registries`,
`rollup_subnational`, `compute_monthly_deltas`, and `infer_hazard` so legacy
tests and notebooks can import the same public API while the connector evolves.

**Header guarantees:** Setting `RESOLVER_SKIP_DTM=1` or `RESOLVER_SKIP_EMDAT=1`
returns immediately but still writes header-only CSVs to
`resolver/staging/dtm_displacement.csv` and `resolver/staging/emdat_pa.csv`
respectively so downstream checks never see missing files.

## ACLED — real connector

- **Endpoint:** Configurable via `ingestion/config/acled.yml` (default `https://api.acleddata.com`). Auth requires
  `ACLED_TOKEN`; override the base with `ACLED_BASE`.
- **Window & paging:** Pulls the last `window_days` (default 450) so that the previous 12 months of battle fatalities are
  available for the onset rule. `ACLED_MAX_LIMIT`, `RESOLVER_MAX_PAGES`, and `RESOLVER_MAX_RESULTS` gate pagination.
- **Conflict fatalities:** All event types are summed per ISO3 × month. Rows emit `metric=fatalities`, `unit=persons`, and
  default to `hazard_code=ACE` (armed_conflict_escalation).
- **Onset detection:** When a country’s battle fatalities over the previous 12 months are `< 25` and the current month is
  `≥ 25`, the hazard switches to `hazard_code=ACO` (armed_conflict_onset). Each row records the inputs in
  `definition_text`/`method`, e.g. `Onset rule inputs: prev12m=24, current_month=30`.
- **Civil unrest:** Counts `event_type ∈ {Protests, Riots}` as `metric=events`, `unit=events`. Optional participant totals
  parse `notes` via a regex when `participants.enabled` or `ACLED_PARSE_PARTICIPANTS=1` (aggregate `sum|median`).
- **Output contract:** `as_of_date=YYYY-MM`, deterministic IDs (`<ISO3>-ACLED-<hazard>-<metric>-YYYY-MM-<digest>`),
  `publisher="ACLED"`, `source_type="other"`, `doc_title="ACLED monthly aggregation"`,
  `method="ACLED; monthly-first; fatality sum across all event types; unrest events=Protests+Riots; onset rule applied"`.
- **Fail-soft:** `RESOLVER_SKIP_ACLED=1` or any runtime error writes a header-only `staging/acled.csv`, keeping CI green.

## WHO Public Health Emergencies — real connector

- **Phase 1 sources:** Config-driven CSV/XLSX feeds listed in `ingestion/config/who_phe.yml` (WHO direct or HDX mirrors).
- **Metric:** Monthly-first **new cases** per ISO3 mapped to `hazard_code=PHE`, `metric=affected`, `unit=persons`.
- **Series semantics:** Daily/weekly incident series are summed to month start; cumulative series are converted to month-over-month deltas (first month dropped unless `WHO_PHE_ALLOW_FIRST_MONTH=1`).
- **PIN expansion:** Optional `WHO_PHE_PIN_RATIO=<float>` emits parallel `metric=in_need` rows using an integer-rounded ratio of cases.
- **Output:** Deterministic `event_id` (`<iso3>-WHO-phe-<metric>-YYYY-MM-<digest>`), `publisher="WHO"`, `source_type="official"`, `method="WHO PHE; monthly-first; …"` with frequency/delta hints.
- **Env toggles:**
  - `RESOLVER_SKIP_WHO=1` — skip connector (writes header-only CSV).
  - `WHO_PHE_ALLOW_FIRST_MONTH=1` — retain the first month from cumulative series instead of dropping it.
  - `WHO_PHE_PIN_RATIO=0.2` (example) — emit `in_need` rows alongside `affected`.
  - `RESOLVER_MAX_RESULTS=<int>` / `RESOLVER_DEBUG=1` follow the shared ingestion conventions.

## HDX (CKAN) — real connector

- **Endpoint:** CKAN Action API (`/api/3/action/package_search`) for dataset discovery, plus resource download URLs (CSV/XLSX).
- **Selection:** iterates ISO3s from `data/countries.csv`, filtering to HNO/HRP topics and preferring HXL-tagged resources.
- **Metrics:** extracts **People in Need** monthly first (`metric=in_need`). Falls back to **People Affected** monthly when PIN is absent.
- **Time handling:** requires month granularity (`YYYY-MM`). Annual rows are emitted only when `ALLOW_ANNUAL_FALLBACK=1`.
- **Hazards:** keyword lexicon maps dataset/resource metadata to `resolver/data/shocks.csv`. Ambiguity defaults to `multi`.
- **Output:** canonical `staging/hdx.csv` with deterministic IDs, `unit=persons`, `publisher="HDX (CKAN)"`, and fail-soft header-only writes.
- **Env toggles:**
  - `RESOLVER_SKIP_HDX=1` — skip connector (writes header-only CSV).
  - `ALLOW_ANNUAL_FALLBACK=1` — permit yearly totals when no monthly data is present (default 0).
  - `RESOLVER_MAX_RESULTS=<int>` — cap emitted rows (shared with other connectors).
  - `RESOLVER_DEBUG=1` — verbose logging for troubleshooting.
  - `HDX_BASE=<url>` — override base URL (default `https://data.humdata.org`).
  - `RELIEFWEB_APPNAME` — optional User-Agent override reused from the ReliefWeb connector.
  - `HDX_ALLOW_PERCENT=1` + `HDX_DENOMINATOR_FILE` — opt-in percent→people conversion using the shared WorldPop denominators.

## WorldPop population denominators — real connector

- **Endpoint:** Configurable CSV mirrors (default template provided in `ingestion/config/worldpop.yml`) exposing national population totals by ISO3 and year.
- **Outputs:**
  - `staging/worldpop.csv` — canonical staging snapshot for auditing the latest fetch.
  - `data/population.csv` — versioned denominator table (`iso3,year,population,source,product,download_date,source_url,notes`).
- **Semantics:** Upserts the most recent year per ISO3 (plus `WORLDPOP_YEARS_BACK` historic years) with overwrite-by-key semantics on reruns.
- **Env toggles:**
  - `RESOLVER_SKIP_WORLDPOP=1` — emit header-only CSVs (no network calls) while
    touching both `data/population.csv` and `staging/worldpop_denominators.csv` so header
    checks remain deterministic.
  - `WORLDPOP_PRODUCT` — choose dataset variant (`un_adj_unconstrained` default; `un_adj_constrained` supported).
  - `WORLDPOP_YEARS_BACK` — fetch additional previous years (default `0`).
  - `WORLDPOP_URL_TEMPLATE` — override the configured URL pattern (handy for mirrors/tests).
  - `RESOLVER_DEBUG=1` — verbose logging and fetch diagnostics.
- **Consumers:** Shared by `WFP mVAM` and (opt-in) `IPC` / `HDX` (via `IPC_ALLOW_PERCENT=1` / `HDX_ALLOW_PERCENT=1`) to turn prevalence into people counts when datasets lack denominators.

## WFP mVAM — real connector

- **Config-driven:** Sources defined in `ingestion/config/wfp_mvam.yml`; supports CSV/XLSX/JSON exports with optional HXL tags.
- **Monthly-first:** Daily/weekly series are averaged to the month before conversion. Example: two February IFC readings of
  `10%` and `12%` with a 100,000-person population average to `11%`, yielding `round(0.11 * 100000) = 11,000` people in need.
- **Percent → people:** Prefer direct people columns. When only prevalence is present, the connector converts the monthly mean (%)
  using dataset-provided populations or the shared WorldPop denominators (`resolver/data/population.csv`). Fallback notes (e.g.,
  `WorldPop un_adj_unconstrained year=2023 (fallback for 2024)`) are written into both `definition_text` and `method` for
  auditability.
- **Outputs:** Emits national **stock** (`series_semantics=stock`) and optional **incident** (`incident` delta) streams with
  deterministic IDs per ISO3/hazard/month. Negative deltas are clipped to zero.
- **Shocks:** Keyword lexicon maps drivers/tags to drought, economic crisis, conflict, flood, or `MULTI` (Multi-driver Food
  Insecurity) when ambiguous.
- **Env toggles:** `RESOLVER_SKIP_WFP_MVAM=1`, `WFP_MVAM_ALLOW_PERCENT` (default `1`), `WFP_MVAM_STOCK`, `WFP_MVAM_INCIDENT`,
  `WFP_MVAM_INCLUDE_FIRST_MONTH_DELTA`, `WFP_MVAM_DENOMINATOR_FILE`, `WFP_MVAM_INDICATOR_PRIORITY`, `WORLDPOP_PRODUCT` (shared
  denominator label hint).

## Source notes (what each adds)

- **EM-DAT** — standardized disaster records and “people affected”; lagged but consistent.
- **GDACS** — near-real-time alerts and modeled impact for hydro-meteo hazards.
- **Copernicus EMS / UNOSAT** — activation footprints, damage/exposure mapping (good PA proxies).
- **HDX (CKAN)** — dataset discovery hub; many country datasets flow here.
- **WorldPop** — annual national population denominators to support percent→people conversions.
- **ACLED / UCDP** — conflict event data (drivers/attribution context; not PIN).
- **FEWS NET** — early warning analyses; aligns with IPC phases for DR/EC.
- **WFP mVAM** — market/price/food security indicators (context for EC/DR).
- **Gov NDMA** — national sitreps; often earliest official “people affected”.

> **Reminder:** Stubs emit plausible demo rows only. Final **resolution** still follows A2 precedence (PIN preferred; PA proxy), and Tier-2/3 sources should not override Tier-1.
