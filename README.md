# Metaculus Forecasting Bot

An advanced forecasting bot for Metaculus that leverages ensemble learning with multiple state-of-the-art LLMs and comprehensive research integration to predict future events.

## Overview

includes:
- **Model Ensembling**: Uses GPT-5, o3, and Sonnet 4 for diverse prediction perspectives
- **Research Integration**: AskNews API with Perplexity fallback for real-time information gathering
- **Advanced Aggregation**: Multiple aggregation strategies including mean, median, and stacking approaches
- **Robust Pipeline**: Comprehensive question processing, research, reasoning, and prediction extraction
- **Numeric/Continuous Question Enhancement**: e.g. PCHIP interpolation (thanks Panshul), tail spreading
- **Prompt Improvements**: obviously  
- **Benchmarking on MC and Numeric Q's**: not just binary  

## Quick Start

### Prerequisites
- Python 3.11+ with conda and poetry
- Required API keys (see Configuration section)

### Setup
1. **Clone and navigate to the repository**
```bash
git clone <repo-url>
cd metaculus-bot
```

2. **Set up conda environment**
```bash
conda create -n metaculus-bot python=3.11
conda activate metaculus-bot
```

3. **Install dependencies**
```bash
make install
# or: conda run -n metaculus-bot poetry install
```

4. **Configure environment**
```bash
cp .env.template .env
# Edit .env with your API keys (see Configuration section)
```

5. **Run the bot**
```bash
make run
# or: conda run -n metaculus-bot poetry run python main.py
```

## Core Architecture

### Main Components
- **`main.py`**: Primary bot implementation using the `forecasting-tools` framework
- **`community_benchmark.py`**: Benchmarking CLI and Streamlit UI for performance evaluation
- **`main_with_no_framework.py`**: Minimal dependencies variant for lightweight usage
- **`metaculus_bot/`**: Core utilities and configurations

### Key Modules
- **`llm_configs.py`**: LLM ensemble configuration and model settings
- **`research_providers.py`**: AskNews and search integration
- **`aggregation_strategies.py`**: Multiple prediction aggregation methods
- **`prompts.py`**: Specialized prompts for different question types
- **`numeric_*.py`**: Numeric question processing and validation

## Usage Examples

### Basic Forecasting
```bash
# Run the bot on current Metaculus questions
make run

# Run with specific question filtering
python main.py --filter-type binary --max-questions 10
```

### Benchmarking
```bash
# Quick smoke test (1 question)
make benchmark_run_smoke_test_binary

# Small benchmark (12 mixed questions) 
make benchmark_run_small

# Large benchmark (100 mixed questions)
make benchmark_run_large

# Display benchmark results
make benchmark_display
```

### Correlation Analysis & Model Filtering

You can analyze correlations and recompute ensembles from prior runs without re-forecasting. Simple substring-based filters let you include or exclude models in the analysis.

Examples:

```bash
# Analyze the most recent benchmark file, excluding Grok and Gemini
PYTHONPATH=. ~/miniconda3/envs/metaculus-bot/bin/python analyze_correlations.py "$(ls -t benchmarks/benchmarks_*.jsonl | head -1)" \
  --exclude-models grok-4 gemini-2.5-pro

# Analyze a directory while excluding models
python analyze_correlations.py benchmarks/ --exclude-models grok-4 gemini-2.5-pro

# Include-only a subset (mutually exclusive with --exclude-models)
python analyze_correlations.py benchmarks/ --include-models qwen3-235b o3

# Apply filters to the built-in post-run analysis
python community_benchmark.py --mode run --num-questions 30 --mixed \
  --exclude-models grok-4 gemini-2.5-pro
```

Notes:
- Matching is substring-only, case-insensitive (no regex or space/hyphen normalization). For example, `grok-4` matches `openrouter/x-ai/grok-4`, but `grok 4` will not.
- Filters apply before computing correlation matrices, model stats, and ensemble search. The generated report includes a “Filters Applied” section.

### Testing
```bash
# Run all tests
make test

# Run specific test file
conda run -n metaculus-bot PYTHONPATH=. poetry run pytest tests/test_specific.py
```

## Configuration

### Required Environment Variables
Create a `.env` file based on `.env.template`:

```bash
# Metaculus API
METACULUS_TOKEN=your_metaculus_token

# Research APIs
ASKNEWS_CLIENT_ID=your_asknews_client_id
ASKNEWS_CLIENT_SECRET=your_asknews_secret
PERPLEXITY_API_KEY=your_perplexity_key
EXA_API_KEY=your_exa_key

# LLM APIs (via OpenRouter)
OPENROUTER_API_KEY=your_openrouter_key
```

### Model Configuration
Models are configured in `metaculus_bot/llm_configs.py`:
- **Primary models**: GPT-5, o3, Sonnet 4 for forecasting
- **Research**: AskNews with Perplexity backup
- **Provider**: OpenRouter for consistent API access

## Development

### Code Quality
```bash
# Lint code
make lint

# Format code
make format

# Install pre-commit hooks
make precommit_install

# Run pre-commit on all files
make precommit_all
```

### Makefile Commands
- `make install` - Install dependencies via conda + poetry
- `make test` - Run pytest suite
- `make run` - Run the forecasting bot
- `make lint` - Run Ruff linting
- `make format` - Format code with Ruff
- `make benchmark_*` - Various benchmarking options

### Testing Philosophy
- Focus on end-to-end integration tests for the forecasting pipeline
- Test core aggregation logic and API integrations
- All tests must pass before PRs
- Use `pytest` with async support for LLM testing

## Repository Structure

```
metaculus-bot/
├── main.py                     # Primary bot implementation
├── community_benchmark.py      # Benchmarking system
├── main_with_no_framework.py   # Minimal variant
├── metaculus_bot/              # Core utilities
│   ├── llm_configs.py         # Model ensemble configuration
│   ├── research_providers.py   # Research integration
│   ├── aggregation_strategies.py # Prediction aggregation
│   ├── prompts.py             # Question-specific prompts
│   └── numeric_*.py           # Numeric processing modules
├── tests/                      # Test suite
├── .github/workflows/          # CI automation
├── AGENTS.md                   # Detailed coding guidelines
└── Makefile                    # Development commands
```

## Framework Integration

This project heavily uses the [`forecasting-tools`](forecasting_tools_readme.md) framework:
- `GeneralLlm` for model interfaces
- `MetaculusApi` for platform integration
- Question types: `BinaryQuestion`, `NumericQuestion`, `MultipleChoiceQuestion`
- Prediction types: `ReasonedPrediction`, `BinaryPrediction`, etc.
- Research: `AskNewsSearcher`, `SmartSearcher`

## Additional Resources

- **[AGENTS.md](AGENTS.md)**: Comprehensive coding guidelines and repository standards
- **[starter_guide.md](starter_guide.md)**: Original template setup instructions
- **[forecasting_tools_readme.md](forecasting_tools_readme.md)**: Framework documentation

## Environment Notes

- **Conda environment**: `metaculus-bot`
- **Python version**: 3.11+
- **Code formatting**: Ruff with 120-character line length
- **Testing**: Pytest with async support
- **Development**: WSL2 environment with zsh terminal
