# Simple forecasting bot

This is a very simple forecasting bot that uses an LLM to forecast on a Metaculus tournament. It lists all questions from the tournament, uses Perplexity.ai to search for up-to-date news about the questions, and then uses ChatGPT to make forecasts.

You need to set these secrets and environment variables:

- `METACULUS_TOKEN` - register your bot to get a token [here](https://www.metaculus.com/aib/)
- `TOURNAMENT_ID` - the ID of the tournament your bot should forecast on
- `OPENAI_API_KEY` - used to access ChatGPT and make forecasts on the questions
- `PERPLEXITY_API_KEY` - used to search for up-to-date information about the questions

It uses GitHub Actions to schedule it for running daily (you can see more about that in [daily_run.yaml](.github/workflows/daily_run.yaml))