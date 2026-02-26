# Integrations

This folder contains example scripts that integrate third-party tools and SDKs with the Metaculus forecasting bot template.

## Install

Integration dependencies are in an optional poetry group. Install them with:

```bash
poetry install --with integrations
```

## Available integrations

### LightningRod Evaluation

**[main_lightningrod_eval.py](main_lightningrod_eval.py)** — Uses the [LightningRod SDK](https://github.com/LightningRodAI/lightningrod-python-sdk) to generate forecasting questions from news. 

Lightning Rod is a composable system for turning raw, timestamped information into model-ready forecasting datasets. Each step is a focused data transform (seed retrieval, question generation, rollouts, context, labeling, etc), these transforms can be chained into reproducible pipelines that operate over structured tables. In practice, this means you can start from sources like news, filings, or domain documents and reliably produce forward-looking questions with standardized outputs (prompt, parser/reward metadata, and resolved answers when available), without hand-labeling every example.

For AI forecasting, this is useful for scaling both evaluation and training data. You can generate large, domain-specific benchmark sets that mirror real forecasting workflows (questions asked at time t with only information available at time t), then score models or forecasters against eventual outcomes. The same pipeline can also produce training corpora for calibration or reasoning improvements, enabling faster iteration on question quality, horizon selection, and label confidence while keeping the process transparent, versioned, and repeatable.

1. Get a `LIGHTNINGROD_API_KEY` at [dashboard.lightningrod.ai](https://dashboard.lightningrod.ai)
2. Add it to your `.env` file
3. Run:
```bash
poetry run python integrations/main_lightningrod_eval.py --max-questions 10
```

## Add your own

Have a tool or SDK that could help with forecasting? Add it here and open a PR!
