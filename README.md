# Simple Metaculus forecasting bot

This repository contains a simple bot meant to get you started with creating your own bot for the [AI Forecasting Tournament](https://www.metaculus.com/aib/).


## Getting started

### Setting up the repository
Fork this repository (click the "Fork" button in the top right corner). You don't have to fork this repository. But if you do, you can make use of the github actions to run your own bot automatically.

Clone the repository. Find your terminal and run the following commands:
```bash
git clone https://github.com/Metaculus/metac-bot
```

If you forked the repository first, you have to replace the url in the `git clone` command with the url to your fork. Just go to your forked repository and copy the url from the address bar in the browser.

### Installing dependencies
Make sure you have python and [poetry](https://python-poetry.org/docs/#installing-with-pipx) installed (poetry is a python package manager).

Inside the terminal, go to the directory you cloned the repository into and run the following command:
```bash
poetry install
```
to install all required dependencies.

### Setting environment variables

#### Locally
Running the bot requires various environment variables. If you run the bot locally, the easiest way to set them is to create a file called `.env` in the root directory of the repository and add the variables in the following format:
```bash
METACULUS_TOKEN=1234567890 # register your bot to get a here: https://www.metaculus.com/aib/
OPENAI_API_KEY=1234567890
PERPLEXITY_API_KEY=1234567890 # optional, if you want to use perplexity.ai
ANTHROPIC_API_KEY=1234567890
```
#### Github Actions
If you want to automate running the bot using github actions, you have to set the environment variables in the github repository settings.
Go to (Settings -> Secrets and variables -> Actions). Set API keys as secrets and the tournament id and API base URL as variables.

For this simple bot, `TOURNAMENT_ID` and `API_BASE_URL` are simply hard-coded in the script and can be changed in the code itself.

## Running the bot

To run the simple bot, execute the following command in your terminal:
```bash
poetry run python simple-forecast-bot.py
```
Make sure to set the environment variables as described above and to set the parameters in the code to your liking. In particular, to submit predictions, make sure that `submit_predictions` is set to `True`.

## Automating the bot using Github Actions

Github can automatically run code in a repository. To that end, you need to fork this repository. You also need to set the secrets and environment variables in the github repository settings as explained above.

Automation is handled in the `.github/workflows/` folder.

The `pr_check.yaml` file is responsible for triggering a test run every time a pull request is made to the main branch. This is useful for development and testing.

The `daily_run_simple_bot.yaml` file runs the simple bot every day (note that since `submit_predictions` is set to `False` in the script by default, no predictions will actually be posted). The `daily_run_simple_bot.yaml` file contains various comments and explanations. You should be able to simply copy this file and modify it to run your own bot.
