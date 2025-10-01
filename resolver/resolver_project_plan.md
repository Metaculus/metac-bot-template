**Resolver - Project Plan**

**Repository:** Spagbot_metac-bot/resolver  
**Owner:** Forecast Ops / Data Eng (Resolver)  
**Status:** Active (Phase H3: UNHCR connectors)  
**Primary metrics:** People in Need (**PIN**) preferred; People Affected (**PA**) fallback  
**Timezone of record:** Europe/Istanbul (UTC+3)

**1) Vision & Goals**

Build a reliable, auditable **database-first** pipeline that ingests humanitarian impact data (PIN/PA) across >130 countries and forecastable shocks, normalizes to a canonical schema, and automatically prepares **resolution-ready** facts for forecasting questions.

**Success Criteria**

- ≥95% successful daily runs (nightly CI green).
- ≥90% of targeted countries & shocks represented monthly within SLA.
- ≤24h lag for routine sources; ≤72h lag for complex/slow sources.
- Zero silent failures: every connector **always** writes a CSV (header-only as fail-soft).
- Reproducible state via **append-only** artifacts, snapshots, and deterministic IDs.

**2) Scope**

**In-scope (current phase)**

- Sources: IFRC GO (Admin v2), ReliefWeb v2 (with WAF handling), UNHCR Refugee Statistics API (annual, month-aware IDs), **UNHCR ODP** (monthly arrivals), and staged additions (OCHA/HDX, IOM DTM, WHO, FEWS NET).
- Shocks (forecastable): Flood, Drought, Tropical Cyclone, Heat Wave, Armed Conflict (onset/escalation/cessation), **Displacement Influx (DI)**, Civil Unrest, Economic Crisis, Public Health Emergency.
- Countries: global coverage using countries.csv (name + ISO3).

**Out of scope (for now)**

- Non-forecastable hazards (e.g., earthquakes) except as historical context.
- Proprietary/credentialed portals unless explicitly configured via secrets.

**3) Operating Model**

- **Data flow** (nightly & on PRs):
  - ingestion/\*\_client.py → resolver/staging/\*.csv (fail-soft headers if empty).
  - tools/export_facts.py → resolver/exports/facts.csv / .parquet.
  - tools/validate_facts.py → schema & content checks (units, enums).
  - tools/precedence_engine.py → cutoff, lags, dedupe → exports/resolved\*.
  - review/make_review_queue.py → review/review_queue.csv.
  - **Auto-commit**: daily state + month-end snapshot (Istanbul).
- **Idempotence**: reruns are safe; deterministic IDs; append-only outputs with monthly snapshots.
- **Human-in-the-loop**: manual override YAML/CSV (review decisions) applied via review/apply_review_overrides.py.

**4) Repository Layout (high level)**

resolver/

data/

countries.csv # country_name, iso3

shocks.csv # hazard_code, hazard_label, hazard_class

ingestion/

config/ # per-source YAMLs (base_url, fields, windows)

ifrc_go_client.py

reliefweb_client.py

unhcr_client.py # refugee statistics API (annual with month-aware IDs)

unhcr_odp_client.py # ODP monthly arrivals (discovered JSON)

run_all_stubs.py

README.md

staging/ # per-source outputs (always present)

\*.csv

tools/

export_facts.py

validate_facts.py

precedence_engine.py

schedule_gate.py

freeze_snapshot.py

write_repo_state.py

exports/

facts.csv

facts.parquet

resolved.csv

resolved.jsonl

resolved_diagnostics.csv

review/

review_queue.csv

decisions_example.csv

apply_review_overrides.py

tests/

test_connectors_headers.py # hermetic header tests

test_unhcr_odp_header.py

snapshots/

YYYY-MM/manifest.json

YYYY-MM/facts.parquet

**5) Canonical Data Contract (staging → facts)**

**Staging CSV columns** (per connector; always present, even if empty):

event_id, country_name, iso3,

hazard_code, hazard_label, hazard_class,

metric, value, unit,

as_of_date, publication_date,

publisher, source_type, source_url, doc_title,

definition_text, method, confidence,

revision, ingested_at

**Facts export**: merges staging sources into a single table; normalizes types, units, enums.

**Precedence/Lag rules** (initial defaults):

- Prefer **PIN** over **PA** when both exist for same (country, hazard, as_of).
- Prefer authoritative agency feeds over media; prefer structured numeric fields over text-extracted numbers.
- Publication lag & cutoff enforced (as_of ≤ cutoff, publication ≤ cutoff + source lag).

**6) CI/CD & Automation**

**Workflow** .github/workflows/resolver-ci.yml:

- Triggers: PRs touching resolver/\*\*; nightly schedule; manual dispatch.
- Steps: setup Python, pip -r resolver/requirements-dev.txt, run connectors, export/validate, precedence, review, tests, **commit artifacts**.
- **Branch-aware push**: PR job pushes to PR head branch; nightly pushes to default branch.
- **Concurrency**: cancel in-progress for same ref.
- **Snapshots**: last day of month (Istanbul) → snapshots/YYYY-MM.

**Hermetic tests**:

- Set RESOLVER_SKIP_\*=1 in tests; each connector writes header-only CSV; no network required.

**7) Environments & Secrets**

**Environment variables**

- Debug & caps: RESOLVER_DEBUG=1, RESOLVER_MAX_PAGES, RESOLVER_MAX_RESULTS, RESOLVER_DEBUG_EVERY.
- Source toggles: RESOLVER_SKIP_RELIEFWEB, RESOLVER_SKIP_IFRCGO, RESOLVER_SKIP_UNHCR, RESOLVER_SKIP_UNHCR_ODP.
- ReliefWeb: RELIEFWEB_APPNAME.
- ODP: ODP_SITUATION_PATH (default /en/situations/europe-sea-arrivals).

**Secrets (GitHub)**

- Add only if a source demands tokens; keep scopes minimal.
- Never commit credentials.

**8) Roles & RACI**

| **Area** | **Responsible** | **Accountable** | **Consulted** | **Informed** |
| --- | --- | --- | --- | --- |
| Data contracts & validation | Data Eng | Project Lead | Forecast Ops, Analysts | All |
| Connectors & configs | Data Eng | Project Lead | Source SMEs | All |
| Precedence & lags | Forecast Ops | Project Lead | Data Eng, Analysts | All |
| CI, snapshots, repo hygiene | Data Eng | Project Lead | DevOps | All |
| Review overrides process | Forecast Ops | Project Lead | Country Focal Points | All |

**9) Milestones & Timeline**

**A. Foundations (done / living docs)**

- A1 Charter & guardrails; A2 Resolution policy; A3 Canonical schema; A4 Roadmap & SLAs.

**B. Pipeline MVP (done)**

- B1 Exporter/Validator; B2 Precedence engine; B3 CI nightly & snapshots (Istanbul month end); B4 Hermetic tests.

**C. Baseline Sources (in progress)**

- C1 IFRC GO Admin v2 (created/updated window, details, caps) ✅
- C2 ReliefWeb v2 (POST filters, WAF fail-soft, appname) ✅
- C3 UNHCR API (asylum-applications annual; month-aware IDs) ✅
- C4 **UNHCR ODP** (monthly arrivals discovery) ✅

**D. Priority Additions (next 2-4 weeks)**

- D1 OCHA/HDX (Operational datasets; PIN where available).
- D2 IOM DTM (displacement/flow monitoring monthly).
- D3 WHO (Public Health Emergency cases, standardized units).
- D4 FEWS NET (IPC/CH phases; food insecurity proxy).
- D5 Governance docs & reviewer playbook (override policy) + dashboards.

**E. Hardening & Scale (rolling)**

- E1 Source-level SLAs & alerting (failed/empty sources).
- E2 Country coverage QA (gaps report).
- E3 PIN extraction improvements (NER + rules).
- E4 Backfills (snapshots by month, replayable).

**10) Acceptance Criteria (per release)**

- **Data contract**: All staging files exist and conform to header; validate_facts.py passes.
- **Coverage**: >80% of targeted country × hazard cells have ≥1 fact in the last 60 days (by source group).
- **Latency**: Median publication_date ≤ 48h from event month end (for monthly sources).
- **Observability**: CI logs surfaced; failure causes attributed per connector; review queue generated.
- **Reproducibility**: Given a snapshots/YYYY-MM manifest, facts.parquet is byte-for-byte stable.

**11) Quality & Risk Controls**

- **Fail-soft** connectors: always write header-only CSV on errors (no pipeline stalls).
- **Deterministic IDs**: SHA-256 short digests; month-aware where applicable to avoid dedup collisions.
- **Timezones**: Istanbul for business logic & snapshots; store timestamps in ISO UTC.
- **WAF/Proxy**: ReliefWeb 202 challenge detected; skip via env in CI; retry/backoff elsewhere.
- **Schema drift**: header tests fail fast; exporters map NaN correctly (fill before cast).
- **Review integrity**: status alignment fixed when rows dropped; overrides documented and auditable.
- **Git hygiene**: rebase abort on failure; artifacts auto-committed with \[skip ci\] to avoid loops.

**12) Connector Roadmap (detail)**

| **Source** | **Metric** | **Cadence** | **Status** | **Notes** |
| --- | --- | --- | --- | --- |
| IFRC GO v2 | PA/PIN | Weekly | Live | countries_details, OR window (created/updated), caps |
| ReliefWeb v2 | PA/PIN | Daily | Live | POST filters only; WAF detection; skip switch |
| UNHCR API | DI  | Annual\* | Live | asylum-applications, year\[\], month-aware IDs when month present |
| UNHCR ODP | DI  | Monthly | Live | Discover JSON links from location pages; sea arrivals |
| OCHA/HDX | PIN | Monthly | Next | Datasets vary; standardize PIN schema + definitions |
| IOM DTM | DI/PA | Monthly | Next | Flow Monitoring, DTM displacement; country feeds |
| WHO | PHE | Weekly | Next | Cases/epi; unit guards (persons_cases) |
| FEWS NET | Food | Monthly | Next | IPC/CH phases mapping to "needs" proxy |

\* The UNHCR API is documented as yearly. We reconcile monthly via ODP (monthly widgets). If monthly parameters become documented in the API, update config (granularity: month) and requests.

**13) Runbooks**

**Daily/Nightly (CI)**

- Ensure secrets & env toggles set (RESOLVER_SKIP_RELIEFWEB=1 until allowlisted).
- Inspect CI logs; WAF/HTTP errors should still produce header-only CSVs.
- Review resolver/exports/resolved_diagnostics.csv for precedence outcomes.
- Triage resolver/review/review_queue.csv; apply overrides if needed.

**Local Dev**

python -m pip install -r resolver/requirements-dev.txt

RESOLVER_DEBUG=1 python resolver/ingestion/run_all_stubs.py

python resolver/tools/export_facts.py --in resolver/staging --out resolver/exports

python resolver/tools/validate_facts.py --facts resolver/exports/facts.csv

python resolver/tools/precedence_engine.py --facts resolver/exports/facts.csv --cutoff YYYY-MM-DD

python resolver/review/make_review_queue.py

python -m pytest resolver/tests -q

**14) Governance & Change Control**

- **PRs** required for schema changes; include data examples and migration notes.
- **Connector changes** must include: updated config, README notes, and header test updates.
- **SLA breaches** (missed run or empty source 2× in a row) → create issue with root cause & remediation plan.

**15) Glossary (selected)**

- **PIN**: People in Need - individuals requiring humanitarian assistance due to a specified shock.
- **PA**: People Affected - broader population impacted, not all necessarily in need.
- **DI**: Displacement Influx - cross-border arrivals into a country (monthly for ODP; annual for API).
- **As-of**: Date the metric applies to (e.g., month mid-point for monthly series).
- **Publication date**: When the source published/updated the figure.

**16) Open Questions / Future Work**

- Documented **monthly** parameters for UNHCR Refugee Statistics API (if/when available).
- PIN standardization across OCHA/HDX datasets (definitions vary).
- Better **country alias** handling (renames, territories).
- **Dashboards**: Streamlit/GitHub Pages for status & coverage.
- **Alerting**: Slack/Email on CI failure, empty sources, or unexpected deltas.

**Changelog (living)**

- **H3**: UNHCR API fixed to year\[\]; deterministic, month-aware IDs; UNHCR ODP monthly connector added; CI installs from requirements-dev.txt.
- **H2**: IFRC GO OR windowing (created/updated), details expansion; ReliefWeb POST-only dataset; WAF fail-soft; hermetic tests.
- **A/B**: Canonical schema, exporter/validator, precedence engine, review queue, snapshots, Istanbul time policy.

**End of plan**