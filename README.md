# Simple Metaculus forecasting bot
This repository contains a simple bot meant to get you started with creating your own bot for the AI Forecasting Tournament. Go to https://www.metaculus.com/aib/ for more info and tournament rules.

In this project are 2 files:
- **main.py**: Our recommended template option that uses [forecasting-tools](https://github.com/Metaculus/forecasting-tools) package to handle a lot of stuff in the background for you (such as API calls). We will update the package, thus allowing you to gain new features with minimal changes to your code.
- **main_with_no_framework.py**: A copy of main.py but implemented with minimal dependencies. Useful if you want a more custom approach.

Join the conversation about bot creation, get support, and follow updates on the [Metaculus Discord](https://discord.com/invite/NJgCC2nDfh) 'build a forecasting bot' channel.

## 30min Video Tutorial
[![Watch the tutorial](https://cdn.loom.com/sessions/thumbnails/fc3c1a643b984a15b510647d8f760685-42b452e1ab7d2afa-full-play.gif)](https://www.loom.com/share/fc3c1a643b984a15b510647d8f760685?sid=29b502e0-cf64-421e-82c0-3a78451159ed)

If you run into trouble, reach out to `ben [at] metaculus [.com]`


## Quick start -> Fork and use Github Actions
The easiest way to use this repo is to fork it, enable github workflow/actions, and then set repository secrets. Then your bot will run every 30min, pick up new questions, and forecast on them. Automation is handled in the `.github/workflows/` folder. The `daily_run_simple_bot.yaml` file runs the simple bot every 30 min and will skip questions it has already forecasted on.

1) **Fork the repository**: Go to the [repository](https://github.com/Metaculus/metac-bot-template) and click 'fork'.
2) **Set secrets**: Go to `Settings -> Secrets and variables -> Actions -> New respository secret` and set API keys/Tokens as secrets. You will want to set your METACULUS_TOKEN. This will be used to post questions to Metaculus, and so you can use our OpenAI/Anthropic proxy (reach out to `ben [at] metaculus [.com]` with your bot description to apply for credits. See the relevant section below).
3) **Enable Actions**: Go to 'Actions' then click 'Enable'. Then go to the 'Regularly forecast new questions' workflow, and click 'Enable'. To test if the workflow is working, click 'Run workflow', choose the main branch, then click the green 'Run workflow' button. This will check for new questions and forecast only on ones it has not yet successfully forecast on.

The bot should just work as is at this point. You can disable the workflow by clicking `Actions > Regularly forecast new questions > Triple dots > disable workflow`

## Getting your Metaculus Token
To get a bot account and your API Token:
1) Go to https://metaculus.com/aib
2) Click "Log Out" if you are using your personal account
3) Click "Create a Bot Account"
4) Create your account
5) Go back to https://metaculus.com/aib
6) Click 'Show My Token'

If your regular Metaculus account uses Gmail, you can create a separate bot account while keeping your existing email by adding a '+bot' before the @ symbol. For example, if your email is 'youremail@gmail.com', you can use 'youremail+bot1@gmail.com' for your bot account.

## Search Provider API Keys

### Getting AskNews Setup
Metaculus is collaborating with AskNews to give free access for internet searches. Each registered bot builder gets 3k calls per month, 9k calls total for the tournament (please note that latest news requests (48 hours back) are 1 call and archive news requests are 5 calls). Bots have access to the /news endpoint only. To sign up:
1. Make an account on AskNews (if you have not yet, https://my.asknews.app)
2. Send the email address associated with your AskNews account to the email `rob [at] asknews [.app]` (or DM `@drafty` in Discord)
3. In that email also send the username of your Metaculus bot
4. AskNews will make sure you have free calls and your account is ready to go for you to make API keys and get going
5. Generate your `ASKNEWS_CLIENT_ID` and `ASKNEWS_SECRET` and add that to the .env
6. Run the AskNewsSearcher from the forecasting-tools repo or use the AskNews SDK python package

Your account will be active for the duration of the tournament. There is only one account allowed per participant.

### Getting Perplexity Set Up
Perplexity works as an internet powered LLM, and costs half a cent per search (if you pick the right model) plus token costs. It is less customizable but generally cheaper.
1. Create an account on the free tier at www.perplexity.ai
2. Go to https://www.perplexity.ai/settings/account
3. Click "API" in the top bar
4. Click "Generate" in the "API Keys" section
5. Add funds to your account with the 'Buy Credits' button
6. Add it to the .env as `PERPLEXITY_API_KEY=your-key-here`

### Getting Exa Set Up
Exa is closer to a more traditional search provider. Exa takes in a search query and a list of filters and returns a list of websites. Each site returned can have scraped text, semantic higlights, AI summary, and more. By putting GPT on top of Exa, you can recreate Perplexity with more control. An implementation of this is available in the `SmartSearcher` of the `forecasting-tools` python package. Each Exa search costs half a cent per search plus a tenth of a cent per 'text-content' requested per site requested. Content items include: highlights from a source, summary of a source, or full text.
1. Make an account with Exa at Exa.ai
2. Go to https://dashboard.exa.ai/playground
3. Click on "API Keys" in the left sidebar
4. Create a new key
5. Go to 'Billing' in the left sidebar and add funds to your acount with the 'Top Up Balance'
6. Add it to the .env as `EXA_API_KEY=your-key-here`

### Other Search
Here are some other unvetted but interesting options for search and website reading:
- Tavily
- Google Search API
- crawl4ai
- Firecrawl
- Playwright

## Accessing the Metaculus LLM Proxy
OpenAI and Anthropic have given a generous donation of credits for bot builders in the tournament which we are providing through an llm proxy.

To get credits assigned to your model choices (or if you need renewed credits from a previous quarter), please send an email to `ben [at] metaculus [.com]` with the below:
* The username of your bot
* A couple paragraph description of how your existing bot works, or what you plan to build
* An estimate of how much budget/tokens you might productively use
* Your preferred Anthropic/OpenAI model(s) and how you want the budget distributed between them (there is budget distributed to each individual model name rather than to your account on whole)

Metaculus will add new OpenAI and Anthropic completion models to the proxy as they come out. If you want to use a new model, please send us an email with the model you desire, and how much budget you want removed from one model and transferred to another. Alternatively, if you have a new idea that needs more support, pitch it to us, and we can add give additional credits. Reach out if you run out.

Visit [this page](href="https://www.notion.so/metaculus/OpenAI-and-Anthropic-credits-0e1f7bf8c8a248e4a38da8758cc04de4") for instructions on how to call the Metaculus proxy directly. 

You can also use the `forecasting-tools` package to call the proxy. To do this, call `await forecasting-tools.GeneralLlm(model="metaculus/{openai_or_anthropic_model_name}").invoke(prompt)`. You will need METACULUS_TOKEN set in your .env file and have already had credits assigned to your account and model choice. GeneralLlm is a wrapper around the litellm package which provides one API for every major model and provider and can be used for other providers like Gemini, XAI, or OpenRouter. For more information about how to use GeneralLlm/litellm see [forecasting-tools](https://github.com/Metaculus/forecasting-tools) and [litellm](https://github.com/BerriAI/litellm)


## Run the bot locally
Clone the repository. Find your terminal and run the following commands:
```bash
git clone https://github.com/Metaculus/metac-bot-template.git
```

If you forked the repository first, you have to replace the url in the `git clone` command with the url to your fork. Just go to your forked repository and copy the url from the address bar in the browser.

### Installing dependencies
Make sure you have python and [poetry](https://python-poetry.org/docs/#installing-with-pipx) installed (poetry is a python package manager).

If you don't have poetry installed run the below:
```bash
sudo apt update -y
sudo apt install -y pipx
pipx install poetry

# Optional
poetry config virtualenvs.in-project true
```


Inside the terminal, go to the directory you cloned the repository into and run the following command:
```bash
poetry install
```
to install all required dependencies.

### Setting environment variables

Running the bot requires various environment variables. If you run the bot locally, the easiest way to set them is to create a file called `.env` in the root directory of the repository (copy the `.env.template`).

### Running the bot

To test the simple bot, execute the following command in your terminal:
```bash
poetry run python main.py --mode quarterly_cup
```
Make sure to set the environment variables as described above and to set the parameters in the code to your liking. In particular, to submit predictions, make sure that `submit_predictions` is set to `True` (it is set to `True` by default in main.py).

## Ideas for bot improvements
Below are some ideas for making a novel bot. Consider using the Benchmarker from the forecasting-tools repo.
- Finetuned LLM on Metaculus Data: Create an optimized prompt (using DSPY or a similar toolset) and/or a fine-tuned LLM using all past Metaculus data. The thought is that this will train the LLM to be well-calibrated on real-life questions.
- Dataset explorer: Create a tool that can find if there are datasets or graphs related to a question online, download them if they exist, and then run data science on them to answer a question.
- Question decomposer: A tool that takes a complex question and breaks it down into simpler questions to answer those instead
- Meta-Forecast Researcher: A tool that searches all major prediction markets, prediction aggregators, and possibly thought leaders to find relevant forecasts, and then combines them into an assessment for the current question (see [Metaforecast](https://metaforecast.org/).
- Base rate researcher: Create a tool to find accurate base rates. There is an experimental version [here](https://forecasting-tools.streamlit.app/base-rate-generator) in [forecasting-tools](https://github.com/Metaculus/forecasting-tools) that works 50% of the time.
- Key factors researcher: Incorporate our experimental [key factors researcher](https://forecasting-tools.streamlit.app/key-factors) to find higher significance key factors for a given question.
- Monte Carlo Simulations: Experiment with combining some tools to run effective Monte Carlo simulations. This could include experimenting with combining Squiggle with the question decomposer.
- Adding personality diversity, LLM diversity, and other variations: Have GPT come up with a number of different ‘expert personalities’ that it runs the forecasting bot with and then aggregates the median. Additionally, run the bot on different LLMs and see if the median of different LLMs improves the forecast. Finally, try simulating up to hundreds of personalities/LLM combinations to create large diverse crowds. Each individual could have a backstory, thinking process, biases they are resistant to, etc. This will ideally improve accuracy and give more useful bot reasoning outputs to help humans reading the output consider things from multiple angles.
- Worldbuilding: Have GPT world build different future scenarios and then forecast all the different parts of those scenarios. It then would choose the most likely future world. In addition to a forecast, descriptions of future ‘worlds’ are created. This can take inspiration from Feinman paths.
- Consistency Forecasting: Forecast many tangential questions all at once (in a single prompt).
- Extremize & Calibrate Predictions: Using the historical performance of a bot, adjust forecasts to be better calibrated. For instance, if predictions of 30% from the bot actually happen 40% of the time, then transform predictions of 30% to 40%.
- Assigning points to evidence: Starting with some ideas from a [blog post from Ozzie Gooen](https://forum.effectivealtruism.org/posts/mrAZFnEjsQAQPJvLh/using-points-to-rate-different-kinds-of-evidence), you could experiment with assigning ‘points’ to major types of evidence and having GPT categorize the evidence it finds related to a forecast so that the ‘total points’ can be calculated. This can then be turned into a forecast, and potentially optimized using machine learning on past Metaculus data.
- Search provider benchmark: Run bots using different combinations of search providers (e.g. Google, Bing, Exa.ai, Tavily, AskNews, Perplexity, etc) and search filters (e.g. only recent data, sites with a certain search rank, etc) and see if any specific one is better than others, or if using multiple of them makes a difference.
- Timeline researcher: Make a tool that can take a niche topic and make a timeline for all major and minor events relevant to that topic.
