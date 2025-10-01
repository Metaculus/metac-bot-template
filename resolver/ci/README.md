# Resolver CI

This directory documents the resolver continuous integration workflows.

## Remote-first commits

CI now commits outputs directly to the repo:

- **PR runs:** `resolver/state/pr/<PR_NUMBER>/...`
- **Nightly runs:** `resolver/state/daily/<YYYY-MM-DD>/...`
- **Month-end snapshot:** `resolver/snapshots/<YYYY-MM>/{facts.parquet,manifest.json}`

Commits are authored by `github-actions[bot]` and include `[skip ci]` to avoid recursive runs.

### Forked PRs
For security, GitHub disables `GITHUB_TOKEN` write permissions on PRs from forks by default.  
If you expect external contributors, keep the old “upload artifact” step as a fallback or switch to `pull_request_target` with careful path filters and review.
