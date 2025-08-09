# Metaculus Prediction Bot

This repository contains a Python-based bot that interacts with the Metaculus prediction platform. The bot uses the OpenAI API to generate predictions for questions on Metaculus and submits them.

Reference `AGENTS.md` in this directory for my general assistant preferences.

## Key Files and Functionality

*   **`main.py`**: The main entry point for the bot. It orchestrates the process of fetching questions from Metaculus, generating predictions using OpenAI's models, and submitting them.
*   **`main_with_no_framework.py`**: A simplified version of the bot's logic, for demonstration purposes, without the more complex structure of the main script.
*   **`community_benchmark.py`**: This script is used to evaluate the bot's performance by comparing its predictions to the Metaculus community's aggregate predictions.
*   **`pyproject.toml`**: Defines the project's dependencies, which include `metaculus`, `openai`, and `numpy`. It uses Poetry for dependency management.
*   **`.github/workflows/`**: This directory contains GitHub Actions workflows for automating the bot's operation.
    *   `run_bot_on_quarterly_cup.yaml`: Runs the bot on the Metaculus Quarterly Cup.
    *   `run_bot_on_tournament.yaml`: Runs the bot on a Metaculus tournament.
    *   `test_bot.yaml`: A workflow for testing the bot's functionality.
*   **`.env.template`**: A template for the required environment variables, which include API keys for Metaculus and OpenAI.

The package uses a framework (`forecasting_tools`), a python lib which you may have trouble accessing directly due to restrictions; I have pasted a static copy of it in the dir `REFERENCE_COPY_OF_forecasting_tools`, which you may access!

To run anything, first activate the conda env:
```bash
conda activate metaculus-bot
```

Then run them with `poetry` (we're using poetry in a Conda env). e.g.: `conda run -n metaculus-bot poetry lock`

Run tests with e.g. (you might also specify given tests/test files):
```bash
conda run -n metaculus-bot poetry run pytest
```

## Setup and Execution

To run the bot, you need to:

1.  Install the dependencies using Poetry: `poetry install`
2.  Create a `.env` file from the `.env.template` and populate it with your Metaculus and OpenAI API keys.
3.  Run the main script: `poetry run python main.py`

Note that the bot will error out locally if you attempt to run inference! This is expected, since we do not store the relevant API keys locally. You should thoroughly test your code without relying on the antipattern of testing with live API calls. However, you may call this locally anyway as a smoke test.

# Git Repository
- The current working (project) directory is being managed by a git repository.
- When asked to commit changes or prepare a commit, always start by gathering information using shell commands:
  - `git status` to ensure that all relevant files are tracked and staged
  - `git diff HEAD` to review all changes (including unstaged changes) to tracked files in work tree since last commit.
    - `git diff --staged` to review only staged changes when a partial commit makes sense or was requested by the user.
  - `git log -n 3` to review recent commit messages and match their style (verbosity, formatting, signature line, etc.)
- Combine shell commands whenever possible to save time/steps, e.g. `git status && git diff HEAD && git log -n 3`.
- Destructive git operations (e.g. `git add`, `git commit`, `git push`, `git reset`) should be left to the user. Non-destructive operations (e.g. `git diff`, `git show`) are perfectly fine.
- The user prefers to add and commit files to git themself. Keeping git ops nondestructive allows async workflows, and adding files to git async breaks the code review loop.  
