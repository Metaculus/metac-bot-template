Awesome — here’s a minimally updated README you can copy-paste in full. I only changed what was necessary to match the current setup (notably: default CSV path, local install instructions to include Poetry, and the repo layout line for `run_spagbot.py`).

---

What is Spagbot?

Spagbot is an AI forecasting assistant. Its job is to read forecasting questions (like those on Metaculus), gather relevant evidence, and combine the judgments of different models into one final forecast. It is designed to run automatically on GitHub, saving its forecasts and learning from past performance over time.

How it works (conceptually)

Understand the question

Spagbot reads a forecasting question (binary “yes/no,” numeric, or multiple-choice).

It builds a tailored prompt to guide models toward giving a probability or range.

Gather research

It can query external news/search APIs (AskNews, Serper) to pull in up-to-date information.

This gives context to the models so they are not just reasoning in a vacuum.

Ask multiple models

Several large language models (like GPT-4o, Claude, Gemini, Grok) are asked the same question.

Each model returns its best estimate (e.g., “There’s a 35% chance…”).

For strategic, political questions, Spagbot can also run a game-theoretic simulation (GTMC1) that models actors, their interests, and possible bargaining outcomes.

Combine forecasts

All these signals are combined using a Bayesian Monte Carlo method.

This ensures the final probability or distribution is mathematically consistent and reflects the strength of evidence from each source.

Calibrate and improve

Spagbot logs every forecast in forecasts.csv (path is configurable).

When questions resolve, it compares its predictions to reality.

A calibration script then adjusts the weights of different models, giving more influence to those that proved more accurate in similar situations.

This creates a closed learning loop so Spagbot improves over time.

Autonomous operation

In GitHub, Spagbot runs on a schedule (for test questions, the tournament, and the Metaculus Cup).

It automatically commits its logs and calibration updates back to the repository, so no human intervention is needed.

The big picture

Think of Spagbot as a forecasting orchestra conductor:

The models are the musicians, each with their own instrument (style of reasoning).

GTMC1 is the percussion section for power politics.

Bayesian Monte Carlo is the conductor, blending them into harmony.

And the calibration loop is like rehearsals, helping everyone play in tune with reality over time.

Quick Start

1. Local run (one question / test mode)

# (Recommended) use Python 3.11 and a virtual environment

pip install -r requirements.txt

# Alternatively, if you use Poetry:

# 1) Install Poetry ([https://python-poetry.org/](https://python-poetry.org/))

# 2) Run:

# poetry install

# poetry run python run\_spagbot.py --mode test\_questions --limit 1

# Environment variables (see “Configuration” below)

# The only must-have for submission is METACULUS\_TOKEN

export METACULUS\_TOKEN=your\_token\_here

# Run a single test question (no submit)

python run\_spagbot.py --mode test\_questions --limit 1

# Run a single question by post ID (no submit)

python run\_spagbot.py --pid 22427

# Submit (be careful!)

python run\_spagbot.py --mode test\_questions --limit 1 --submit

2. Autonomous runs on GitHub

This repo includes workflows for:

test\_bot\_fresh.yaml (fresh research),

test\_bot\_cached.yaml (use cache),

run\_bot\_on\_tournament.yaml,

run\_bot\_on\_metaculus\_cup.yaml,

calibration\_refresh.yml (periodic update of calibration weights).

Add secrets (see below), push the repo, and Actions will:

run forecasts,

write logs to the repo,

update calibration weights on a schedule,

commit/push changes automatically.

What Spagbot Does (Pipeline)

Select question(s)
Test set, single PID, tournament, or Metaculus Cup.

Research (research.py)
Uses AskNews / Serper to pull context. Results are summarized and logged.

Prompting (prompts.py)
Builds compact, question-type aware prompts (binary, MCQ, numeric).

LLM Ensemble (providers.py, ensemble.py)
Calls multiple LLMs asynchronously, with rate-limit guards and usage/cost capture.

Binary → extracts a single probability p(yes)

MCQ → extracts a probability vector across options

Numeric → extracts P10/P50/P90 quantiles

Game-Theoretic Signal (optional, binary) — GTMC1.py
A reproducible, Bueno de Mesquita/Scholz-style bargaining Monte Carlo using an actor table (position, capability, salience, risk). Output includes:

exceedance\_ge\_50 (preferred probability-like signal),

coalition rate, dispersion, rounds to converge, etc.

Aggregation (aggregate.py + bayes\_mc.py)
Bayesian Monte Carlo fusion of all evidence (LLMs + GTMC1).

Binary: Beta-Binomial update → final p(yes).

MCQ: Dirichlet update → final probability vector.

Numeric: Mixture from quantiles → final P10/P50/P90.

Submission (cli.py)
Submits to Metaculus if --submit is set and METACULUS\_TOKEN is present.
Enforces basic constraints (e.g., numeric quantile ordering).

Logging (io\_logs.py)

Machine-readable master CSV: forecasts.csv (path configurable via FORECASTS\_CSV\_PATH)

Human log per run: forecast\_logs/runs/\<run\_id>.md (or .txt)
In GitHub Actions, logs auto-commit/push unless disabled.

Calibration Loop (update\_calibration.py)
On a schedule (e.g., every 6–24h), reads forecasts.csv, filters resolved questions (and only forecasts made before resolution), computes per-model skill and writes:

calibration\_weights.json (used next run to weight models)

data/calibration\_advice.txt (human-readable notes)
These outputs are committed, so the loop closes autonomously.

Repository Layout (key files)
(run at repo root)
run\_spagbot.py        # Thin wrapper: calls cli.main()

spagbot/
cli.py                # Orchestrates runs, submission, and final log commit
research.py           # AskNews/Serper search + summarization
prompts.py            # Prompt builders per question type
providers.py          # Model registry, rate-limiters, cost estimators
ensemble.py           # Async calls to each LLM + parsing to structured outputs
aggregate.py          # Calls bayes\_mc to fuse LLMs (+ GTMC1) into final forecast
bayes\_mc.py           # Bayesian Monte Carlo updaters (Binary, MCQ, Numeric)
GTMC1.py              # Game-theoretic Monte Carlo with reproducibility guardrails
io\_logs.py            # Canonical logging to forecast\_logs/ + auto-commit
seen\_guard.py         # Simple JSONL-based “skip duplicates” registry (optional)
update\_calibration.py # Builds class-conditional weights from resolutions
**init**.py           # Package export list

forecast\_logs/
forecasts.csv         # (if you set FORECASTS\_CSV\_PATH here) otherwise default is repository root
runs/                 # Human-readable per-run logs (.md or .txt)

data/
calibration\_advice.txt  # Friendly notes from the calibration job (auto-created)

gtmc\_logs/
...                    # Optional run-by-run GTMC CSVs/metadata (if enabled)

Configuration

Set these as environment variables (locally or in GitHub Actions). Sensible defaults are used where possible.

Required (for submission)

METACULUS\_TOKEN — your Metaculus API token.

Recommended

FORECASTS\_CSV\_PATH=forecasts.csv
(default; set only if you want a different location)

CALIB\_WEIGHTS\_PATH=calibration\_weights.json

CALIB\_ADVICE\_PATH=data/calibration\_advice.txt

HUMAN\_LOG\_EXT=md (or txt)

LOGS\_BASE\_DIR=forecast\_logs (default)

Git commit behavior (logs & calibration outputs)

In GitHub Actions:

DISABLE\_GIT\_LOGS=false (or unset) → auto-commit/push logs & calibration outputs

Locally:

COMMIT\_LOGS=true → will also commit/push from your machine (optional)

Optional:

GIT\_LOG\_MESSAGE="chore(logs): update forecasts & calibration"

GIT\_REMOTE\_NAME=origin

GIT\_BRANCH\_NAME=main (detected automatically; this is a safe fallback)

LLM/API keys (if used in your setup)

OPENROUTER\_API\_KEY, GOOGLE\_API\_KEY, XAI\_API\_KEY, etc.

Research keys: ASKNEWS\_CLIENT\_ID, ASKNEWS\_CLIENT\_SECRET, SERPER\_API\_KEY, etc.

Tip: Put non-secret defaults in .env.example and store secrets in GitHub → Settings → Secrets and variables → Actions.

Running Modes

Test set (no submit by default):
python run\_spagbot.py --mode test\_questions --limit 4

Single question by PID:
python run\_spagbot.py --pid 22427

Tournament / Cup (CI workflows call these internally):

run\_bot\_on\_tournament.yaml

run\_bot\_on\_metaculus\_cup.yaml
Include --submit in those workflows if you want automatic submissions.

Logging & Files Written

Machine CSV: forecasts.csv (canonical, append-only)
Contains per-question details, per-model parsed outputs, final aggregation, timestamps, etc.

Human log: forecast\_logs/runs/<timestamp>\_\<run\_id>.md
Friendly narrative summary: which models ran, research snippets, final forecast.

These are committed automatically from GitHub Actions unless disabled.

Calibration (Autonomous Loop)

Input: forecasts.csv

Logic:

Keep resolved questions only.

Keep only forecasts made before resolution time.

Compute per-model loss by question type and class (e.g., binary vs numeric; topical classes if present).

Build class-conditional weights with shrinkage toward global weights.

Output:

calibration\_weights.json

data/calibration\_advice.txt (friendly notes)

Use: At the start of a run, Spagbot loads calibration\_weights.json to weight ensemble members.

Important: Ensure the data/ folder exists in the repo so calibration\_advice.txt can be written and committed.

Game-Theoretic Module (GTMC1)

Accepts an actor table with columns: name, position, capability, salience, risk\_threshold (scales 0–100 for the first three).

Runs a deterministic Monte Carlo bargaining process (PCG64 RNG seeded from a fingerprint of the table + params).

Outputs:

exceedance\_ge\_50 (preferred probability-like signal when higher axis = “YES”),

median\_of\_final\_medians,

coalition\_rate,

dispersion,

and logs a per-run CSV under gtmc\_logs/ (optional to commit).

You can enable/disable GTMC1 by question type or flags in cli.py.

Aggregation Math (Plain English)

Binary: Treat each model’s p(yes) (and GTMC1’s exceedance) as evidence with a confidence weight. Update a Beta prior; take the posterior mean as final probability (with quantile summaries P10/P50/P90).

MCQ: Treat each model’s probability vector as evidence and update a Dirichlet prior; take the posterior mean vector.

Numeric: Fit a Normal from each model’s P10/P50/P90, sample proportionally to confidence weights, and compute final P10/P50/P90 from the mixture.

Weights come from calibration\_weights.json, updated by the calibration job.

GitHub Actions (What’s Included)

Fresh test runs vs cached test runs

Tournament and Metaculus Cup runners

Calibration refresh (scheduled)

Each workflow:

Sets the environment (paths, commit behavior),

Runs the bot (and/or update\_calibration.py),

Auto-commits any modified files under:

forecast\_logs/,

calibration\_weights.json,

data/ (advice),

(optionally) state files you decide to persist.

If you want duplicate-protection to persist across CI runs, store seen\_guard state inside the repo (e.g., forecast\_logs/state/seen\_forecasts.jsonl) and include it in commits.

Secrets You Must Add (GitHub → Actions)

METACULUS\_TOKEN (for submission)

Any LLM / research API keys you plan to use:

OPENROUTER\_API\_KEY, GOOGLE\_API\_KEY, XAI\_API\_KEY, ASKNEWS\_CLIENT\_ID, ASKNEWS\_CLIENT\_SECRET, SERPER\_API\_KEY, etc.

Troubleshooting

“No calibration changes to commit.”
Normal if no new resolutions or no forecasts before the resolution time.

Numeric submission errors (CDF monotonicity etc.).
The numeric path enforces ordered quantiles (P10 ≤ P50 ≤ P90). If Metaculus still complains, reduce extremely steep distributions (too tiny sigma) or clip the extreme tails. (The current code already guards the common pitfalls.)

Logs not committing locally.
Set COMMIT\_LOGS=true and ensure your git remote/branch are correct.

Advice file write error.
Ensure data/ folder exists and isn’t .gitignore’d.
