# Integrations

This folder contains example scripts that integrate third-party tools and SDKs with the Metaculus forecasting bot template.

## Install

Integration dependencies are in an optional poetry group. Install them with:

```bash
poetry install --with integrations
```

## Available integrations

### LightningRod SDK

**[main_lightningrod_eval.py](main_lightningrod_eval.py)** - Uses the [LightningRod SDK](https://github.com/lightning-rod-labs/lightningrod-python-sdk) to generate forecasting questions from news. 

The **Lightning Rod SDK** is a Python library for generating custom forecasting datasets. It transforms real-world data sources into labeled forecasting samples automatically, using built-in integrations like Google News, or your own documents.

The output is an exportable dataset you can use to benchmark LLMs, or train on to improve calibration and sharpen reasoning. The SDK covers the full pipeline: ingesting sources, generating questions, labeling questions, and scoring against real outcomes. 

Data generation pipelines are fully customizable: including date ranges, question format (e.g. binary, multiple choice), and custom instructions to shape the output. 

The Metaculus community gets **$100 in free credits** with code `METACULUS100`. Create an account at [lightningrod.ai](https://lightningrod.ai/) and explore the [SDK examples](https://github.com/lightning-rod-labs/lightningrod-python-sdk/tree/main/notebooks) to get started.

1. Get a `LIGHTNINGROD_API_KEY` at [dashboard.lightningrod.ai](https://dashboard.lightningrod.ai)
2. Add it to your `.env` file
3. Run:

```bash
poetry run python integrations/main_lightningrod_eval.py
```

## Add your own

Have a tool or SDK that could help with forecasting? Add it here and open a PR!