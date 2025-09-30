# Governance & Audit — Resolver (A3)

## Evidence & Traceability
- Save the original PDF/HTML as an **artifact** (private bucket); store `artifact_id` + `artifact_sha256`.
- Capture a **verbatim snippet** (±200 chars) around the figure in `definition_text`.
- Record `method=api|scrape|manual` and `confidence=high|med|low`.

## Versioning & Snapshots
- **Append-only** facts: new totals are **new rows** with `revision = previous + 1`.
- **Monthly freeze**: at **23:59 Europe/Istanbul** on the **last day** of each month:  
  - Write `snapshots/YYYY-MM/facts.parquet` + `manifest.json` (`created_at_utc`, `source_commit_sha`).
  - **Scoring uses snapshots** for that month; live views use current facts.

## Source Precedence & Conflict
- Precedence (highest→lowest):  
  `inter_agency_plan > ifrc_or_gov_sitrep > un_cluster_snapshot > reputable_ingo_un > media_discovery_only`
- **One total only** — never sum across agencies.
- **Conflict rule**: if eligible figures differ by **>20%** → choose higher precedence; if same tier, use **newest as_of** then **latest publication**. Keep the alternative in `alt_value` + `alt_source_url` and explain in `precedence_decision`.

## Attribution & Scope
- A record must **explicitly link** the number to the hazard episode (per policy).  
- Use country totals; sub-national figures only if they roll up (or are the official national total).

## Quality Controls
- Schema checks for required fields.
- Range checks on `value`.
- Date sanity (as_of ≤ publication ≤ now).
- Registry checks: `iso3` and `hazard_code` **must** exist in the CSV registries.
- Outliers (e.g., `value` > national population) → flag `confidence=low` and route to review.

## Roles & Access
- **Ingestion bot**: append facts + upload artifacts.  
- **Analyst**: review/override low-confidence records (add a note).  
- **Resolver**: read-only; returns single number + citation.  
- **Public dashboard**: read filtered; no artifacts, no PII.

## Licensing & Compliance
- Only official/publicly accessible documents.  
- Store `publisher` and any license/terms when available.  
- Respect robots.txt for scrapes.

## Retention
- Facts: indefinite (trend analysis).  
- Artifacts: ≥24 months; cold-store after 12 months.

## Monitoring
- Source freshness dashboard (last success per source).  
- Ingestion counts by source/day; error rate; % low-confidence.  
- Alert on missing freezes and on conflict spikes.
