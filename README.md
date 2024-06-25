# Simple forecasting bot

This is a very simple forecasting bot which uses an LLM to forecase on a Metaculus tournament.
It lists all questions from the tournament, uses perplexity.ai to search up to date news about the question and then uses ChatGPT to forecast on it.

You have to set these secrets and environment variables
- `METACULUS_TOKEN` - register your bot to get a token [here](https://www.metaculus.com/aib/)
- `TOURNAMENT_ID` - the ID of the tournament your bot should forecast on
- `OPENAI_API_KEY` - used to use ChatGPT and forecast on the question
- `PERPLEXITY_API_KEY` - optional, used to search up to date information about the question.