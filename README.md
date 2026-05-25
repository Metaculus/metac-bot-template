# Simple Metaculus forecasting bot
This repository contains a simple bot meant to get you started with creating your own bot for the AI Forecasting Tournament. Go to https://www.metaculus.com/futureeval/participate/ for more info and tournament rules (and then go to the  "Getting Started" section of our [resources](https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/#want-to-join-the-ai-forecasting-benchmark) page).

**Brand new to this?** You can get a working bot running in about 5 minutes without writing a single line of code — just fork this repo, paste two API keys into GitHub, and click "Run workflow". See **[Quick start](#quick-start--fork-and-use-github-actions)** below.

In this project are 2 files:
- **main.py**: Our recommended template option that uses the [forecasting-tools](https://github.com/Metaculus/forecasting-tools) package to handle a lot of stuff in the background for you (such as API calls). We will update the package, thus allowing you to gain new features with minimal changes to your code.
- **main_with_no_framework.py**: A copy of main.py but implemented with minimal dependencies. Useful if you want a more custom approach.


Join the conversation about bot creation, get support, and follow updates on the [Metaculus Discord](https://discord.com/invite/NJgCC2nDfh) 'build a forecasting bot' channel.

## 30min Video Tutorial
This tutorial shows you how to set up our template bot so you can start forecasting in the tournament.

[![Watch the tutorial](https://cdn.loom.com/sessions/thumbnails/fc3c1a643b984a15b510647d8f760685-42b452e1ab7d2afa-full-play.gif)](https://www.loom.com/share/fc3c1a643b984a15b510647d8f760685?sid=29b502e0-cf64-421e-82c0-3a78451159ed)

If you run into trouble, reach out to `ben [at] metaculus [.com]`


## Quick start -> Fork and use Github Actions
The easiest way to use this repo is to fork it, paste in two API keys, and click "Run workflow". After that, the bot will keep forecasting on new questions automatically every 20 minutes — no local setup needed.

1) **Fork the repository** — go to the [repository](https://github.com/Metaculus/metac-bot-template) and click **Fork** in the top right.
2) **Add your two API keys as repository secrets** — in your fork, go to `Settings → Secrets and variables → Actions → New repository secret`. Add these two (names must match exactly, all caps):
   - **`METACULUS_TOKEN`** — create one at https://www.metaculus.com/futureeval/participate/ (see the [resources page](https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/#creating-your-bot-account-and-metaculus-token) if you get stuck).
   - **`OPENROUTER_API_KEY`** — get free credits via [this form](https://forms.gle/aQdYMq9Pisrf1v7d8), or make your own key on [OpenRouter](https://openrouter.ai/). You can also use `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `PERPLEXITY_API_KEY`, `ASKNEWS_SECRET`, etc. — these all work out of the box if you set them.
3) **Enable Actions** — click the `Actions` tab, then click `I understand my workflows, go ahead and enable them`.
4) **Run the test workflow to confirm everything works** — go to `Actions → Test Bot → Run workflow → Run workflow` (green button). This forecasts on whatever's currently open in the [bot-testing-area tournament](https://www.metaculus.com/tournament/bot-testing-area/) so you can verify your setup posts forecasts to Metaculus end-to-end. Once the run finishes (~3–5 min), check your bot's profile on Metaculus to confirm the forecasts landed.
5) **You're done!** The `Forecast on new AI tournament questions` workflow is already enabled and will run every 20 minutes, picking up any new tournament questions and skipping ones it has already forecast on.

To pause your bot, go to `Actions → Forecast on new AI tournament questions → ... (top right) → Disable workflow`.

### Testing your changes against the GitHub Actions workflow
You can run any workflow against any branch — no need to merge to `main` first, and no need to fork if you have push access to this repo.

1. Push your branch to GitHub: `git push origin <your-branch>`.
2. In the repo's Actions tab, pick the workflow you want to run (e.g. `Test Bot`) and click **Run workflow** (top right).
3. Use the **"Use workflow from"** dropdown to select your branch instead of `main`, then click the green **Run workflow** button.

The runner checks out your branch and uses the repo's existing secrets — those are scoped to the repo, not the branch, so they work for any branch in the same repo. This works for all three workflows.

## API Keys
Instructions for getting your METACULUS_TOKEN, OPENROUTER_API_KEY, or optional search provider API keys (AskNews, Exa, Perplexity, etc) are listed on the "Getting Started" section of the [resources](https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/#want-to-join-the-ai-forecasting-benchmark) page.

## Changing the Github automation
To run a different script under the same workflows, edit the `poetry run python main.py` line in the appropriate file under `.github/workflows/` and replace `main.py` with your script. The workflows that exist:
- `test_bot.yaml` — manual-trigger smoke test against the bot-testing-area tournament.
- `run_bot_on_tournament.yaml` — every 20 min on the live AIB tournament + MiniBench.
- `run_bot_on_metaculus_cup.yaml` — every 2 days on the Metaculus Cup.

**To run `main_with_no_framework.py` via GitHub Actions instead of `main.py`:** open the workflow file you want and change `poetry run python main.py` to `poetry run python main_with_no_framework.py`. That's the only change required.

## Editing in GitHub UI
Remember that you can edit a bot non locally by clicking on a file in Github, and then clicking the 'Edit this file' button. Whether you develop locally or not, when making edits, attempt to do things that you think others have not tried, as this will help further innovation in the field more than doing something that has already been done. Feel free to ask about what has or has not been tried in the Discord, see [other bot's self-descriptions](https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/#what-are-other-bots-doing), or read bot's [open source code](https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/#open-source-bots).

## Run/Edit the bot locally
Local development is optional — most new users can run the bot entirely from GitHub Actions (see [Quick start](#quick-start--fork-and-use-github-actions)). Set up locally only if you want faster iteration on your prompts/code.

### 1. Clone the repository
```bash
git clone https://github.com/Metaculus/metac-bot-template.git
cd metac-bot-template
```
If you've already forked the repo, replace the URL with your fork's URL (copy it from your fork's page in the browser).

### 2. Install Python 3.11+ and Poetry
You need:
- **Python 3.11 or newer** — get it from [python.org](https://www.python.org/downloads/) (or your OS package manager / `pyenv` / whatever you prefer).
- **Poetry** — see Poetry's [install docs](https://python-poetry.org/docs/#installation). The `pipx install poetry` route works on macOS, Linux, and Windows.

Confirm both are on your `PATH`:
```bash
python --version    # 3.11.x or higher
poetry --version
```

(Optional, recommended) Keep the virtualenv inside the project directory so your editor picks it up automatically:
```bash
poetry config virtualenvs.in-project true
```

### 3. Install dependencies
From inside the cloned repository:
```bash
poetry install
```

### 4. Set your API keys
Copy the template and fill in your real keys:
```bash
cp .env.template .env
```
Then open `.env` in any text editor and replace each `REPLACE_ME` with your real key. At minimum you need `METACULUS_TOKEN` and one LLM key (`OPENROUTER_API_KEY` is recommended). See the comments inside `.env.template` for where to get each one.

### 5. Run the bot
**First run — smoke-test against the [bot-testing-area tournament](https://www.metaculus.com/tournament/bot-testing-area/):**
```bash
poetry run python main.py --mode test_questions
```
You'll see a one-line startup banner, forecasting progress logs, then a `🎉 Bot submitted N forecast(s)` banner with direct links to each forecast on Metaculus.

**Forecast on live AIB tournament + MiniBench:**
```bash
poetry run python main.py --mode tournament
```

**Forecast on the Metaculus Cup:**
```bash
poetry run python main.py --mode metaculus_cup
```

**Run the no-framework reference implementation instead:**
```bash
poetry run python main_with_no_framework.py
```
This file has no `--mode` flag; it's controlled by the constants at the top of the file (`SUBMIT_PREDICTION`, `USE_EXAMPLE_QUESTIONS`, `TOURNAMENT_ID`, etc.). Flip `USE_EXAMPLE_QUESTIONS = True` to point it at the bot-testing-area tournament instead of the live AIB.

To stop publishing forecasts (dry-run mode):
- `main.py`: set `publish_reports_to_metaculus=False` in the `SummerTemplateBot2026(...)` constructor near the bottom.
- `main_with_no_framework.py`: set `SUBMIT_PREDICTION = False` at the top.

## Example usage of /news and /deepnews:
If you are using AskNews, here is some useful example code.
```python
from asknews_sdk import AsyncAskNewsSDK
import asyncio

"""
More information available here:
https://docs.asknews.app/en/news
https://docs.asknews.app/en/deepnews

Installation:
pip install asknews
"""

client_id = ""
client_secret = ""

ask = AsyncAskNewsSDK(
    client_id=client_id,
    client_secret=client_secret,
    scopes=["chat", "news", "stories", "analytics"],
)

# /news endpoint example
async def search_news(query):

  hot_response = await ask.news.search_news(
      query=query, # your natural language query
      n_articles=5, # control the number of articles to include in the context
      return_type="both",
      strategy="latest news" # enforces looking at the latest news only
  )

  print(hot_response.as_string)

  # get context from the "historical" database that contains a news archive going back to 2023
  historical_response = await ask.news.search_news(
      query=query,
      n_articles=10,
      return_type="both",
      strategy="news knowledge" # looks for relevant news within the past 60 days
  )

  print(historical_response.as_string)

# /deepnews endpoint example:
async def deep_research(
    query, sources, model, search_depth=2, max_depth=2
):

    response = await ask.chat.get_deep_news(
        messages=[{"role": "user", "content": query}],
        search_depth=search_depth,
        max_depth=max_depth,
        sources=sources,
        stream=False,
        return_sources=False,
        model=model,
        inline_citations="numbered"
    )

    print(response)


if __name__ == "__main__":
    query = "What is the TAM of the global market for electric vehicles in 2025? With your final report, please report the TAM in USD using the tags <TAM> ... </TAM>"

    sources = ["asknews"]
    model = "deepseek-basic"
    search_depth = 2
    max_depth = 2
    asyncio.run(
        deep_research(
            query, sources, model, search_depth, max_depth
        )
    )

    asyncio.run(search_news(query))
```

Some tips for DeepNews:

You will get tags in your response, including:

<think> </think>
<asknews_search> </asknews_search>
<final_response> </final_response>

These tags are likely useful for extracting the pieces that you need for your pipeline. For example, if you don't want to include all the thinking/searching, you could just extract <final_response> </final_response>


## Integrations

The **[integrations/](integrations/)** folder contains example scripts that integrate third-party tools with the bot template. 

See the [integrations README](integrations/README.md) for available integrations and how to add your own.

## Ideas for bot improvements
You can find some ideas of what you can do to improve this template by taking a look at what other bots have done [here](https://www.metaculus.com/notebooks/43497/what-are-other-bots-doing/). You can also look at research done by Metaculus and the field in the [research section](https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/#research-reports-and-overview-of-the-field) of the bot resources page. Asking an LLM to read through everything and give ideas may be a decent place to start.
