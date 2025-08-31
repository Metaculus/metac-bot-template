# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Metaculus AI forecasting bot template for the AI Forecasting Tournament. The repository provides two main implementations:
- `main.py`: Uses the `forecasting-tools` package (recommended)
- `main_with_no_framework.py`: Minimal dependencies implementation

## Development Commands

### Setup
```bash
uv sync
```

### Running the Bot

#### Framework Version (main.py)
```bash
# Test on example questions
uv run python main.py --mode test_questions

# Run on tournament questions
uv run python main.py --mode tournament

# Run on quarterly cup
uv run python main.py --mode quarterly_cup
```

#### No Framework Version (main_with_no_framework.py)
```bash
# Run with minimal dependencies (configure constants at top of file)
uv run python main_with_no_framework.py
```

### Benchmarking
```bash
# Run benchmark against community predictions
uv run python community_benchmark.py --mode run

# View benchmark results in UI
uv run streamlit run community_benchmark.py
```

## Environment Configuration

Copy `.env.template` to `.env` and configure:
- `METACULUS_TOKEN`: Required - Get from https://metaculus.com/aib
- Search provider keys (at least one): `ASKNEWS_CLIENT_ID`/`ASKNEWS_SECRET`, `PERPLEXITY_API_KEY`, `EXA_API_KEY`
- LLM API keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`

## Architecture

### Core Classes
- `TemplateForecaster(ForecastBot)`: Main bot implementation in `main.py`
  - `run_research()`: Gathers information using search providers
  - `_run_forecast_on_binary/multiple_choice/numeric()`: Generates predictions
  - Uses `forecasting-tools` package for API interactions and utilities

### Search Providers (Priority Order)
1. AskNews (free for tournament participants)
2. Exa + SmartSearcher 
3. Perplexity

### Bot Flow
1. Load questions from Metaculus API
2. For each question: run research → generate predictions → aggregate → submit
3. Supports concurrent processing with rate limiting

## Key Files
- `main.py`: Main bot using forecasting-tools framework
- `main_with_no_framework.py`: Standalone implementation
- `community_benchmark.py`: Benchmarking against community predictions
- `pyproject.toml`: uv dependency management
- `.github/workflows/`: GitHub Actions for automated forecasting

## Testing
- Use `--mode test_questions` for development
- Benchmark against community predictions before deployment
- GitHub Actions run bot every 30 minutes on tournament questions