# Metaculus forecasting bot

Predicting future events with LLMs

forked from metaculus starter  
rough changelog:  
- model changes, using gpt5, gemini 2.5 pro, and r1-0528; gemini 2.5 flash for summarization; for now pplx for new, soon asknews  
- model ensembling  
- aggregation changes (e.g. use mean to reduce cost of bagging)  
- prompt changes  
- added basic test suite  

## Repository Guidelines
Coding agents: for coding standards, local commands, testing, and PR expectations, see [AGENTS.md — Repository Guidelines](AGENTS.md#repository-guidelines).
See also: `starter_guide.md` for some basic info from the template's creators and `forecasting_tools_readme.md` for info on the forecasting-tools pkg used to interact w/ Metaculus.

## Formatting & Linting
- Lint: `make lint` (Ruff check)
- Format: `make format` (Ruff format + autofix)
- Pre-commit hooks:
  - Install: `make precommit_install`
  - Run on staged files: `make precommit`
  - Run on all files: `make precommit_all`

Ruff is configured in `pyproject.toml` with `line-length = 120`. The linter ignores `E501` since formatting is handled by `ruff-format`.

# Archived starter info
See starter_guide.md for verbose setup instructions.
