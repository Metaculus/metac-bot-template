# Integrations

This folder contains example scripts that integrate third-party tools and SDKs with the Metaculus forecasting bot template.

## Install

Integration dependencies are in an optional uv dependency group. Passing `--group integrations` to `uv run` (as in the commands below) installs them on first use.

## Available integrations

### LightningRod SDK

**[main_lightningrod_eval.py](main_lightningrod_eval.py)** - Uses the [LightningRod SDK](https://github.com/lightning-rod-labs/lightningrod-python-sdk) to generate forecasting questions from news. 

The **Lightning Rod SDK** is a Python library for generating custom forecasting datasets. It transforms real-world data sources into labeled forecasting samples automatically, using built-in integrations like Google News, or your own documents.

The output is an exportable dataset you can use to benchmark LLMs, or train on to improve calibration and sharpen reasoning. The SDK covers the full pipeline: ingesting sources, generating questions, labeling questions, and scoring against real outcomes. 

Data generation pipelines are fully customizable: including date ranges, question format (e.g. binary, multiple choice), and custom instructions to shape the output. 

Create an account at [lightningrod.ai](https://lightningrod.ai/) and explore the [SDK examples](https://github.com/lightning-rod-labs/lightningrod-python-sdk/tree/main/notebooks) to get started. Check the LightningRod site or reach out to them for any promotions available to the Metaculus community.

1. Get a `LIGHTNINGROD_API_KEY` at [dashboard.lightningrod.ai](https://dashboard.lightningrod.ai)
2. Add it to your `.env` file
3. Run:

```bash
uv run --group integrations python integrations/main_lightningrod_eval.py
```

### Bot Review

**[metaculus-bot-review](https://github.com/LouisP96/metaculus-bot-review)** - Scores your bot's resolved forecasts and displays its reasoning on specific questions.

It reads the reports your bot published as Metaculus comments, so there is nothing to set up beyond the `METACULUS_TOKEN` you already have. No LLM spend is incurred.

```bash
uv run --group integrations bot-review review --tournament <slug-or-id> --output review.json --summary review.md
uv run --group integrations bot-review review --resolved-since 30    # anything that resolved recently
```

`review.md` gives your rank, how many questions were scored, and the best and worst questions on whichever score the leaderboard uses. `review.json` adds per-question detail, including every forecaster's prediction on every run.

Reasoning text is not in the table. Pull it a piece at a time:

```bash
uv run --group integrations bot-review show <POST_ID> --section research
uv run --group integrations bot-review show <POST_ID> --forecaster R1:F3
```

Two extras ship with this template:

- **[.github/workflows/review_bot.yaml](../.github/workflows/review_bot.yaml)** runs the review weekly and attaches `review.json` and `review.md` to the run. It needs only `METACULUS_TOKEN`. It is off by default: set the repository variable `REVIEW_BOT_ENABLED` to `true` to turn it on (Settings → Secrets and variables → Actions → Variables).
- **[.claude/skills/review-bot/SKILL.md](../.claude/skills/review-bot/SKILL.md)** drives the whole loop with Claude Code and writes up what it finds. Ask it to review how the bot did on a tournament.

## Add your own

Have a tool or SDK that could help with forecasting? Add it here and open a PR!