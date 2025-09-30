# Epic A — Strategy & Guardrails

## Vision
Establish a durable strategy for the **resolver** program so that Spagbot can
confidently update forecast records once questions settle on Metaculus. The epic
covers the early groundwork: aligning on scope, defining the success metrics,
setting access guardrails, and mapping the minimum architecture required for
later engineering sprints.

## Problem Statement
- `forecast_logs/forecasts.csv` exposes resolution fields (`resolved`,
  `resolved_time_iso`, `resolved_outcome_label`, `resolved_value`) that are
  currently empty or populated by ad‑hoc scripts.
- Calibration refreshes depend on accurate resolved outcomes to compute Brier
  and log scores; without these, automated calibration drifts.
- The repository lacks a dedicated module ("resolver") to normalize resolution
  data ingestion, auditing, and propagation into downstream artifacts.

## Objectives
1. **Define scope & stakeholders** – Document who owns the resolver, how it
   interfaces with Metaculus APIs, calibration, and analytics.
2. **Codify guardrails** – Ensure any automation respects rate limits,
   credentials, and write safety for logs.
3. **Outline the north-star architecture** – Identify the components, data
   contracts, and operational runbooks needed for reliable resolution syncing.
4. **Deliver program metrics** – Decide how we will measure success and keep the
   resolver aligned with forecasting outcomes (e.g., % of forecasts resolved
   within SLA, % of logs reconciled).

## Success Criteria
- Program charter reviewed with core contributors (forecast ops, calibration,
  infra).
- Shared glossary describing the lifecycle of a Metaculus question inside
  Spagbot (from forecast to resolution and scoring).
- Documented guardrails ratified in `resolver/` and referenced by future
  engineering epics.
- Initial milestone plan and acceptance criteria ready for implementation
  sprints (Epics B, C, …).

## Guardrails & Constraints
- **Data integrity first**: Resolver scripts must never mutate existing rows in
  place without append-only backups. All destructive actions require dry-runs.
- **Credential hygiene**: Tokens and API keys continue to load from environment
  variables or GitHub secrets. No hard-coded credentials or scopes beyond what
  Spagbot already uses.
- **Idempotent jobs**: Resolver workflows need to be rerunnable. Duplicate runs
  should not double-count scores or corrupt csv history.
- **Observability**: Every automation must emit structured logs (JSON or CSV)
  and human-readable markdown summaries similar to current forecast logs.
- **Human overrides**: Maintain a manual override path (e.g., YAML patch files)
  for edge cases where Metaculus data is ambiguous or delayed.
- **Read-only analytics consumers**: Downstream notebooks/dashboards should
  consume resolver outputs via stable, documented schemas.

## Workstreams
1. **Program Foundations**
   - Draft resolver charter with roles/responsibilities (owner, reviewers,
     on-call person for failed jobs).
   - Capture stakeholder interviews (Metaculus API usage, calibration cadence,
     dashboard needs).
   - Produce glossary and lifecycle diagrams.

2. **Guardrail Specification**
   - Document API usage limits, retry policies, and failure handling.
   - Define data validation rules for resolved outcomes (type checks, range,
     cross-field consistency).
   - Specify audit-trail requirements (e.g., checksum of pre/post rows).

3. **Architecture Blueprint**
   - Map end-to-end flow: ingestion → normalization → storage → scoring →
     calibration refresh.
   - Identify required directories/files (e.g., `resolver/jobs/`,
     `resolver/contracts/`).
   - Outline interfaces with existing modules (`spagbot/cli.py`, calibration
     scripts, dashboards).

4. **Program Metrics & KPIs**
   - Define key metrics (resolution coverage, latency, accuracy, number of
     manual overrides).
   - Establish monitoring dashboard requirements (inputs for Streamlit or
     GitHub summary reports).
   - Draft SLA/SLO tables and escalation paths for breaches.

## Deliverables
- `resolver/charter.md` (future task) capturing ownership, process, cadence.
- `resolver/guardrails.md` enumerating API/data safeguards.
- `resolver/architecture.md` with sequence diagrams and data contracts.
- `resolver/metrics.md` detailing KPIs and reporting cadences.
- Consolidated backlog for follow-up engineering epics (B: ingestion, C:
  normalization, etc.).

## Milestones & Timeline (tentative)
| Milestone | Description | Target |
|-----------|-------------|--------|
| M1 | Kick-off, stakeholder alignment, initial charter draft | Week 1 |
| M2 | Guardrails finalized, validation checklist approved | Week 2 |
| M3 | Architecture blueprint reviewed, dependencies captured | Week 3 |
| M4 | Metrics framework + backlog grooming for Epics B/C | Week 4 |

## Dependencies & Risks
- **Metaculus API stability**: Changes to endpoints or auth flows could break
  ingestion; plan for version pinning.
- **Historical data quality**: Past forecasts may have inconsistent IDs or
  missing metadata, requiring backfill scripts.
- **Team bandwidth**: Coordination with calibration owners may be limited; need
  asynchronous review workflows.
- **Security reviews**: Any new GitHub Action that touches secrets might need
  organization-level approval.

## Open Questions
- Should resolver operate as a standalone CLI entry point or integrate into
  existing `run_spagbot.py` workflows?
- Do we need database storage (SQLite/Parquet) beyond CSV for efficient
  backfills?
- What is the desired cadence (daily, hourly) for resolution syncing during
  tournaments?
- How do we surface manual overrides back to the community dashboard to avoid
  silent corrections?

## Exit Criteria for Epic A
- Consensus on strategy, guardrails, and architecture direction documented in
  `resolver/`.
- Approved backlog for subsequent implementation epics with owners assigned.
- Sign-off from calibration maintainer that success metrics and guardrails meet
  their needs.
