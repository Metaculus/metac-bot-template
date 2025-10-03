# Resolution Policy (A2) — PIN preferred; PA as proxy (aligned to Resolver Hazard Types; NO earthquakes)

> **Applies globally.** All records must carry:
> - `country_name` + `iso3` (from `resolver/data/countries.csv`)
> - `hazard_label` + `hazard_code` + `hazard_class` (from `resolver/data/shocks.csv`)
>
> **Scope note:** Per program decision, **earthquakes and other non-forecastable geophysical hazards are out of scope** for resolution forecasting in this initiative. The hazard registry in `shocks.csv` reflects the **uploaded Resolver Hazard Types.xlsx** and excludes these.

## 1) Purpose
Resolve questions of the form:
- **Preferred:** “By `<DATE>`, how many **People in Need (PIN)** due to hazard `<Y>` in `<COUNTRY>`?”
- **Fallback/Proxy:** “By `<DATE>`, how many **People Affected (PA)** due to hazard `<Y>` in `<COUNTRY>`?”

Outputs must be **deterministic, auditable, and reproducible** from monthly snapshots.

## 2) Scope
- Countries: all in `resolver/data/countries.csv` (Name + ISO3 **required**).
- Hazards: those in `resolver/data/shocks.csv` (Label, Code, Class) derived from the uploaded taxonomy; **earthquakes excluded**.
- Time basis: continuous ingestion; **monthly freeze** (23:59 **Europe/Istanbul** last day of month) for grading.

## 3) Core Definitions
- **PIN** — Inter-agency **People in Need** (HNO/HRP/Flash Appeal/Addendum or OCHA country sitrep).
- **PA** — **People Affected** (broader exposure/impact; not equivalent to PIN).
- **As-of date** — The date the figure explicitly refers to; if absent, use publication date.

## 4) Resolution Order
**Metric preference:** try **PIN**; if unavailable, resolve **PA** and set `proxy_for=PIN`.

**Source precedence (highest → lowest):**
1. **Inter-agency** plans/docs: HNO/HRP/Flash/Addendum, OCHA country sitrep
2. **IFRC GO** (DREF/Emergency Appeal) or **Government NDMA** official sitrep
3. **UN/Cluster** official snapshots (UNHCR, IOM-DTM, WHO, IPC/Health/Food clusters)
4. Reputable **UN/INGO** agency reports
5. **Media/news**: *discovery only* (not final) unless the article embeds/links the official doc, which we also capture

**One total only:** never sum agency numbers; select the **single authoritative** country total for the hazard.

## 5) Cut-off & Freshness
- Accept documents **published on or before** the question cut-off (23:59 **Europe/Istanbul** on `<DATE>`), or with **as-of ≤ cut-off**.
- Allow a **7-day publication lag** if the document clearly states an earlier as-of (≤ cut-off).

## 6) Geography & Attribution
- Country scope = the stated country; sub-national figures may be used **only** if the document provides (or clearly implies) a country total.
- Accept a figure **explicitly attributed** to the hazard `<Y>` (e.g., “flooding linked to Cyclone X outer bands”).
  If multiple drivers exist, the document must tie the number to `<Y>`. Otherwise use proxy with `confidence=low`.

## 7) Conflict Rule
If eligible figures differ by **>20%**:
- Choose the **higher-precedence** source.
- If same tier, pick the **newest as-of** date (then latest publication date).
- Record the discarded value in `alt_value` + `alt_source_url`; add a short note in `precedence_decision`.

### Conflict Onset Rule (ACLED ingestion)
For ACLED-derived conflict monitoring we apply an explicit **conflict onset** rule in addition to standard escalation checks:

- Track **battle fatalities** each month for every ISO3 country code, counting only ACLED events whose `event_type` is in the configurable `onset.battle_event_types` list (default: `Battles`).
- Compute the rolling sum of battle fatalities over the **previous 12 months**, excluding the current month. Missing months are treated as zero so gaps do not break the window.
- When the previous-12-month total is **< 25** battle deaths and the current month records **≥ 25** battle deaths, we create an additional Resolver fact with `hazard_code=ACO` (*Armed Conflict – Onset*).
- The same monthly battle fatality total is emitted as the escalation series (`hazard_code=ACE`). Diagnostics capture the rolling lookback, threshold, and applied configuration (`onset_rule_v1`).
- Threshold and lookback parameters live in `resolver/ingestion/config/acled.yml` (`onset.threshold_battle_deaths` and `onset.lookback_months`) so policy changes are auditable without code edits.

## 8) Proxy Ladder (aligned to the uploaded hazard classes)
> The exact `hazard_class` values come from `resolver/data/shocks.csv`. Use these class rules; refine per sub-type as needed.

- **Hydro-meteorological (e.g., tropical cyclone/typhoon, severe storm, flood, storm surge, heavy rainfall):**  
  Prefer **official “people affected”** from OCHA/IFRC/Gov sitreps when PIN absent. If multiple updates exist, apply precedence + conflict rule. If still missing, use standardized disaster totals from recognized repositories (e.g., EM-DAT) as last resort.  
  → `metric=affected`, `proxy_for=PIN`, `confidence=med`.

- **Drought / Food insecurity:**  
  Use **IPC Phase 3+ population** in the affected footprint when PIN is not yet published.  
  → `metric=affected`, `proxy_for=PIN`, `note=IPC P3+ proxy`, `confidence=med`.

- **Conflict / Violence / Forced displacement:**  
  Use **IOM-DTM new IDPs** or **UNHCR new arrivals/affected** attributable to the escalation window when PIN missing.  
  → `metric=affected` (or `displaced` if explicitly stated), `proxy_for=PIN`, `confidence=med`.

- **Epidemic / Outbreak (incl. WASH-linked):**  
  Use **WHO** “people affected/at-risk” if defined; if only **cases** exist, record `metric=cases`, `unit=persons_cases`, and **do not** coerce to PIN.  
  → `confidence=low|med` depending on definition clarity.

- **Macro-economic / Price shock / Services collapse (if present in taxonomy):**  
  Use **inter-agency/cluster assessments** reporting **PA** or *people requiring assistance*; where absent, use **triangulated official totals** (e.g., targeted populations in cash/food assistance scale-ups) clearly tied to the shock.  
  → `metric=affected`, `proxy_for=PIN`, `confidence=low|med`.

> **Explicit exclusions:** **Earthquakes** (and other geophysical hazards you mark as non-forecastable) are **out of scope**.

Always capture the **definition text verbatim** that describes who is counted.

## 9) Revisions
- Newer official totals **supersede** earlier ones (append a new revision; never destructive updates).
- Resolver returns the **latest eligible** total before cut-off.  
- **Snapshots are immutable** (grading uses the snapshot of that month).

## 10) Audit Requirements (store per record)
- `country_name`, `iso3`, `hazard_label`, `hazard_code`, `hazard_class`
- `metric` (`in_need`/`affected`/`displaced`/`cases`…), `value`, `unit`
- `as_of_date`, `publication_date`
- `publisher`, `source_type`, `source_url`, `doc_title`
- `definition_text` (verbatim line/paragraph around the figure)
- `method` (`api|scrape|manual`), `confidence` (`high|med|low`)
- `revision`, `artifact_id`, `artifact_sha256`
- `notes`, `alt_value`, `alt_source_url`, `proxy_for`, `precedence_decision`

## 11) Worked Examples (no earthquake example)

**Example A — Tropical Cyclone / Typhoon (Hydro-met)**  
- *Country:* Philippines (PHL)  
- *Hazard:* Typhoon (hazard_code per `shocks.csv`)  
- *Preferred:* **PIN** from Flash Appeal/HRP addendum; else **PA** from OCHA/NDMA/IFRC sitrep.  
- *If two sitreps conflict by >20%:* select inter-agency total; record NDMA alternative in `alt_value`.

**Example B — Drought / Food Insecurity (Slow-onset)**  
- *Country:* Ethiopia (ETH)  
- *Hazard:* Drought (hazard_code per `shocks.csv`)  
- *Preferred:* **PIN** from HNO/HRP; if missing at cut-off, use **IPC Phase 3+** population in affected regions (proxy).  
- *Audit:* Store IPC link + Phase 3+ definition; set `proxy_for=PIN`, `confidence=med`.

**Example C — Conflict Escalation / Displacement (Human-induced)**  
- *Country:* Sudan (SDN)  
- *Hazard:* Conflict escalation (hazard_code per `shocks.csv`)  
- *Preferred:* **PIN** from HRP/addendum if issued; else **IOM-DTM new IDPs** explicitly tied to the event window (proxy).  
- *Conflict rule:* If Gov vs DTM differ by >20%, apply precedence; record alternative.

## 12) Machine-Readable Policy (YAML)
```yaml
metric_preference: [in_need, affected]
source_precedence:
  - inter_agency_plan       # HNO/HRP/Flash/Addendum/OCHA sitrep
  - ifrc_or_gov_sitrep
  - un_cluster_snapshot
  - reputable_ingo_un
  - media_discovery_only
conflict_rule:
  threshold_pct: 20
  tie_breaker: latest_as_of
cutoff:
  timezone: Europe/Istanbul
  lag_days_allowed: 7
proxy_ladder:
  hydro_meteorological: ocha_ifrc_gov_affected_else_standard_repo
  drought_food_insecurity: ipc_phase3_plus
  conflict_displacement: iom_dtm_new_idps_or_unhcr_new_arrivals
  epidemic_outbreak: who_defined_affected_or_cases_only
  macroeconomic_services: inter_agency_or_cluster_assessment
exclusions:
  - earthquakes
audit_fields:
  - country_name
  - iso3
  - hazard_label
  - hazard_code
  - hazard_class
  - metric
  - value
  - unit
  - as_of_date
  - publication_date
  - publisher
  - source_type
  - source_url
  - doc_title
  - definition_text
  - method
  - confidence
  - revision
  - artifact_id
  - artifact_sha256
  - notes
  - alt_value
  - alt_source_url
  - proxy_for
  - precedence_decision
```

13) Definition of Done (DoD)

resolver/policy/resolution_policy.md contains the full policy above, aligned to the uploaded hazard list, with earthquakes excluded.

Includes 3 worked examples (none geophysical/earthquake).

Includes the YAML section.
