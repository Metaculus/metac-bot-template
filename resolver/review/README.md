# Review Queue

Human-in-the-loop checks for edge cases before we publish/grade.

## What gets flagged
- Conflicts > 20% (from `exports/resolved_diagnostics.csv`)
- Low confidence (`confidence = low`)
- Media-sourced rows when `metric = in_need`
- Proxy usage (`proxy_for = PIN`)
- Date anomalies (as_of > publication or publication > today)
- Tier risk: selected `precedence_tier` is not top tiers

## Files
- `review_queue.csv` — you edit this file:
  - `analyst_decision` ∈ `keep|override|drop`
  - if `override`, fill `override_value`, `override_source_url` (opt), `override_notes` (opt)

## Workflow
```bash
python resolver/review/make_review_queue.py
# edit resolver/review/review_queue.csv
python resolver/review/apply_review_overrides.py \
  --resolved resolver/exports/resolved.csv \
  --decisions resolver/review/review_queue.csv \
  --out resolver/exports/resolved_reviewed.csv
```
