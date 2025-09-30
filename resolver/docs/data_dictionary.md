# Data Dictionary — Resolver (A3)

This dictionary defines the **facts** records the resolver stores and reads.  
All records **must** include both **country_name + iso3** and **hazard_label + hazard_code + hazard_class** (aligned to the registries in `/resolver/data/`).

## Canonical registries (authoritative)
- `/resolver/data/countries.csv` — columns: `country_name,iso3`
- `/resolver/data/shocks.csv` — columns: `hazard_code,hazard_label,hazard_class`  
  (`hazard_class` ∈ `natural | human-induced | epidemic`; **earthquakes excluded** per scope)

## Table: facts (append-only)
| Field | Type | Req | Description |
|---|---|:--:|---|
| event_id | string | R | Stable ID for a country–hazard–episode–revision |
| country_name | string | R | From countries registry |
| iso3 | string(3) | R | From countries registry |
| hazard_code | string | R | From shocks registry |
| hazard_label | string | R | From shocks registry |
| hazard_class | string | R | `natural|human-induced|epidemic` (from shocks registry) |
| metric | enum | R | `in_need` (PIN) \| `affected` \| `displaced` \| `cases` … |
| value | number | R | Non-negative integer (persons) |
| unit | enum | R | `persons` \| `persons_cases` (for outbreaks) |
| as_of_date | date(YYYY-MM-DD) | R | Date figure refers to |
| publication_date | date | R | Publication date of source |
| publisher | string | R | OCHA, IFRC, NDMA, UNHCR, IOM-DTM, WHO, etc. |
| source_type | enum | R | `appeal|sitrep|gov|cluster|agency|media` |
| source_url | string | R | Canonical URL of doc/portal |
| doc_title | string | R | Title of doc/page |
| definition_text | text | R | Verbatim local definition of who is counted |
| method | enum | R | `api|scrape|manual` |
| confidence | enum | R | `high|med|low` |
| revision | int | R | 1…n (newer supersedes older) |
| artifact_id | string | O | Path/key in object storage |
| artifact_sha256 | string | O | Hash of saved artifact |
| notes | text | O | Free-form notes |
| alt_value | number | O | Kept when conflict rule discards an eligible figure |
| alt_source_url | string | O | Source of alternative value |
| proxy_for | enum/null | O | `PIN` when using proxy (e.g., IPC P3+) |
| precedence_decision | text | O | Brief rationale (e.g., “HNO superset; NDMA older”) |
| ingested_at | datetime(UTC) | R | Insert timestamp |

**Constraints & validation**
- `value >= 0`; if `value > national_population(iso3)` ⇒ flag `confidence=low`.
- `as_of_date <= publication_date <= now`.
- Uniqueness hint (soft): `(iso3, hazard_code, metric, as_of_date, publisher, revision)`.
- `metric='in_need'` only when source explicitly publishes **PIN**.
- For outbreaks where only cases exist, use `metric=cases`, `unit=persons_cases` (do **not** coerce to PIN).
