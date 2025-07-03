# Metaculus Prediction Bot

This repository contains a Python-based bot that interacts with the Metaculus prediction platform. The bot uses the OpenAI API to generate predictions for questions on Metaculus and submits them.

## Key Files and Functionality

*   **`main.py`**: The main entry point for the bot. It orchestrates the process of fetching questions from Metaculus, generating predictions using OpenAI's models, and submitting them.
*   **`main_with_no_framework.py`**: A simplified version of the bot's logic, likely for testing or demonstration purposes, without the more complex structure of the main script.
*   **`community_benchmark.py`**: This script is used to evaluate the bot's performance by comparing its predictions to the Metaculus community's aggregate predictions.
*   **`pyproject.toml`**: Defines the project's dependencies, which include `metaculus`, `openai`, and `numpy`. It uses Poetry for dependency management.
*   **`.github/workflows/`**: This directory contains GitHub Actions workflows for automating the bot's operation.
    *   `run_bot_on_quarterly_cup.yaml`: Runs the bot on the Metaculus Quarterly Cup.
    *   `run_bot_on_tournament.yaml`: Runs the bot on a Metaculus tournament.
    *   `test_bot.yaml`: A workflow for testing the bot's functionality.
*   **`.env.template`**: A template for the required environment variables, which include API keys for Metaculus and OpenAI.

## Setup and Execution

To run the bot, you need to:

1.  Install the dependencies using Poetry: `poetry install`
2.  Create a `.env` file from the `.env.template` and populate it with your Metaculus and OpenAI API keys.
3.  Run the main script: `poetry run python main.py`

This documentation will be available for future reference.
