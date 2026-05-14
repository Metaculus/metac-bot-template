# Psychohistory bootstrap notes (two-repo split)

This repo is the live AIB posting bot (Metaculus template).
Keep psychohistory harness in its own repo for memory/Brier research loops.

## 1) Env setup

```bash
cp .env.template .env
```

Set at least:
- `METACULUS_TOKEN`
- `OPENROUTER_API_KEY` (free-credit path)
- `ASKNEWS_CLIENT_ID`
- `ASKNEWS_SECRET`
- `PSYCHOHISTORY_OUTPUT_DIR` (absolute path in psychohistory repo, e.g. `/Users/darenpalmer/conductor/workspaces/psychohistory-v2/aib-live-next-steps/data/template-runs`)

Compatibility aliases included:
- `METACULUS_API_TOKEN`
- `ASKNEWS_API_KEY`

## 2) Install + smoke test

```bash
poetry install
poetry run python main.py --mode test_questions
```

## 3) Live run + export

```bash
poetry run python main.py --mode tournament
```

When `PSYCHOHISTORY_OUTPUT_DIR` is set, successful forecast reports are appended to:
- `runs-YYYY-MM-DD.jsonl`

If the env var is not set, export is skipped and template behavior is unchanged.

## 4) Sync into psychohistory memory

In psychohistory repo:

```bash
python -m scripts.sync_template_runs \
  --template-output-dir /Users/darenpalmer/conductor/workspaces/psychohistory-v2/aib-live-next-steps/data/template-runs \
  --memory-dir /Users/darenpalmer/conductor/workspaces/psychohistory-v2/aib-live-next-steps/.harness_memory
```

This writes `EpisodicRecord`s into `JsonlMemoryStore` idempotently.

## 5) Resolution/Brier updates

Run sync again after outcomes appear in exported rows (when `resolved_outcome` is present), or enrich outcome fields before sync. The sync script calls `resolve_market()` and writes Brier scores idempotently.

## 6) Known limitations
- Current psychohistory harness runner path is binary-first (`final_p_yes` scalar).
- Summer/FutureEval includes numeric, discrete, and multiple-choice question types.
- For launch, run binary-capable flows first (e.g., bot testing area / MiniBench subsets), then extend posting adapters for non-binary output shapes.

## 7) GitHub Actions (recommended)
- Fork this repo to your account.
- Add secrets in GitHub Actions settings:
  - `METACULUS_TOKEN`
  - `OPENROUTER_API_KEY`
  - `ASKNEWS_CLIENT_ID`
  - `ASKNEWS_SECRET`
- Add variable/secret:
  - `PSYCHOHISTORY_OUTPUT_DIR`
- Enable workflow and run once manually.
