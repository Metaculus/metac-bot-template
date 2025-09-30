# Data Registries — Resolver

Authoritative CSVs that standardize **countries** and **hazards** across the resolver.

## Files
- `countries.csv` — `country_name,iso3`
- `shocks.csv` — `hazard_code,hazard_label,hazard_class`

## Rules
- These registries are the **only** allowed values for `iso3`, `hazard_code`, `hazard_label`, `hazard_class`.
- Update via PRs; keep changes backwards-compatible (additive when possible).
- Per scope: **earthquakes and other non-forecastable geophysical hazards are excluded** from `shocks.csv`.
