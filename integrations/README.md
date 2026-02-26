# Integrations

This folder contains example scripts that integrate third-party tools and SDKs with the Metaculus forecasting bot template.

## Install

Integration dependencies are in an optional poetry group. Install them with:

```bash
poetry install --with integrations
```

## Available integrations

### LightningRod Evaluation

**[main_lightningrod_eval.py](main_lightningrod_eval.py)** — Uses the [LightningRod SDK](https://github.com/LightningRodAI/lightningrod-python-sdk) to generate forecasting questions from news. Generated datasets can be used for fine-tuning or evaluation.  

1. Get a `LIGHTNINGROD_API_KEY` at [dashboard.lightningrod.ai](https://dashboard.lightningrod.ai)
2. Add it to your `.env` file
3. Run:
```bash
poetry run python integrations/main_lightningrod_eval.py --max-questions 10
```

## Add your own

Have a tool or SDK that could help with forecasting? Add it here and open a PR!
