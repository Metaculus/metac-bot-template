
#!/usr/bin/env python

import json

import requests
from decouple import config
import datetime
import re

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import Settings
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI

# Note: To understand this code, it may be easiest to start with the `main()`
# function and read the code that is called from there.


def build_prompt(
        title: str,
        description: str,
        resolution_criteria: str,
        fine_print: str,
        news_info: str | None = None
    ):
    """
    Function to build the prompt using various arguments.
    """

    prompt = f"""
You are a professional forecaster interviewing for a job.

Your interview question is:
{title}

background:
{description}

{resolution_criteria}

{fine_print}

"""

    if news_info:
        prompt += f"""
Your research assistant says:
{news_info}

"""


    prompt += f"""
Today is {datetime.datetime.now().strftime("%Y-%m-%d")}.

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) What the outcome would be if nothing changed.
(c) What you would forecast if there was only a quarter of the time left.
(d) What you would forecast if there was 4x the time left.

You write your rationale and then the last thing you write is your final answer as: "Probability: ZZ%", 0-100
"""

    return prompt

def process_forecast_probabilty(forecast_text: str):
    """
    Extract the forecast probability from the forecast text and clamp it between 1 and 99.
    """
    matches = re.findall(r"(\d+)%", forecast_text)
    if matches:
        # Return the last number found before a '%'
        number = int(matches[-1])
        number = min(99, max(1, number)) # clamp the number between 1 and 99
        return number
    else:
        return None


def list_questions(base_url: str, metac_token: str, tournament_id: int, offset=0, count=10):
    """
    List questions from a specific tournament. This uses the questions
    endpoint and queries it for questions belonging to a specific tournament.

    Parameters:
    -----------
    base_url : str
        the base url of the metaculus API
    metac_token : str
        the token to use for authentication
    tournament_id : int
        the ID of the tournament to list questions from
    offset : int, optional
        the number of questions to skip. This is used for pagination. I.e. if
        offset is 0 and count is 10 then the first 10 questions are returned.
        If offset is 10 and count is 10 then the next 10 questions are returned.
    count : int, optional
        the number of questions to return

    Returns:
    --------
    json
        A list of JSON objects, each containing information for a single question
    """
    # a set of parameters to pass to the questions endpoint
    url_qparams = {
        "limit": count, # the number of questions to return
        "offset": offset, # pagination offset
        "has_group": "false",
        "order_by": "-activity", # order by activity (most recent questions first)
        "forecast_type": "binary", # only binary questions are returned
        "project": tournament_id, # only questions in the specified tournament are returned
        "status": "open", # only open questions are returned
        "format": "json", # return results in json format
        "type": "forecast", # only forecast questions are returned
        "include_description": "true", # include the description in the results
    }
    url = f"{base_url}/questions/" # url for the questions endpoint
    response = requests.get(
        url,
        headers={"Authorization": f"Token {metac_token}"},
        params=url_qparams
    )
    # you can verify what this is doing by looking at
    # https://www.metaculus.com/api2/questions/?format=json&has_group=false&limit=5&offset=0&order_by=-activity&project=3294&status=open&type=forecast
    # in the browser. The URL works as follows:
    # base_url/questions/, then a "?"" before the first url param and then a "&"
    # between additional parameters

    response.raise_for_status()
    data = json.loads(response.content)
    return data["results"]


def call_perplexity(query):
    """
    Make a call to the perplexity API to obtain additional information.

    Parameters:
    -----------
    query : str
        The query to pass to the perplexity API. This is the question we want to
        get information about.

    Returns:
    --------
    str
        The response from the perplexity API.
    """
    url = "https://api.perplexity.ai/chat/completions"
    api_key = config("PERPLEXITY_API_KEY", default="-")
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    payload = {
        "model": "llama-3.1-sonar-large-128k-chat",
        "messages": [
            {
                "role": "system", # this is a system prompt designed to guide the perplexity assistant
                "content": """
You are an assistant to a superforecaster.
The superforecaster will give you a question they intend to forecast on.
To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
You do not produce forecasts yourself.
""",
            },
            {
                "role": "user", # this is the actual prompt we ask the perplexity assistant to answer
                "content": query,
            },
        ],
    }
    response = requests.post(url=url, json=payload, headers=headers)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    print(
        f"\n\nCalled perplexity with:\n----\n{json.dumps(payload)}\n---\n, and got\n:",
        content,
    )
    return content


def get_model(model_name: str):
    """
    Get the appropriate language model based on the provided model name.
    This uses the classes provided by the llama-index library.

    Parameters:
    -----------
    model_name : str
        The name of the model to instantiate. Supported values are:
        "gpt-4o", "gpt-3.5-turbo", "anthropic", "o1-preview"

    Returns:
    --------
    Union[OpenAI, Anthropic, None]
        An instance of the specified model, or None if the model name is not recognized.

    Note:
    -----
    This function relies on environment variables for API keys. These should be
    stored in a file called ".env", which will be accessed using the
    `config` function from the decouple library.
    """

    match model_name:
        case "gpt-4o":
            return OpenAI(
                api_key=config("OPENAI_API_KEY", default=""),
                model=model_name
            )
        case "gpt-3.5-turbo":
            return OpenAI(
                api_key=config("OPENAI_API_KEY", default=""),
                model=model_name
            )
        case "anthropic":
            tokenizer = Anthropic().tokenizer
            Settings.tokenizer = tokenizer
            return Anthropic(
                api_key=config("ANTHROPIC_API_KEY", default=""),
                model="claude-3-5-sonnet-20240620",
            )
        case "o1-preview":
            return OpenAI(
                api_key=config("OPENAI_API_KEY", default=""),
                model=model_name
            )

    return None


def main():
    """
    Main function to run the forecasting bot. This function accesses the questions
    for a given tournament, fetches information about them, and then uses an LLM
    to generate a forecast.

    Parameters:
    -----------
    None. All relevant parameters are devined via environment variables or
    directly in the code.

    Installing dependencies
    ----------------------
    Install poetry: https://python-poetry.org/docs/#installing-with-pipx.
    Then run `poetry install` in your terminal.

    Environment variables
    ----------------------

    You need to have a .env file with all required environment variables.

    Alternatively, if you're running the bot via github actions, you can set
    the environment variables in the repository settings.
    (Settings -> Secrets and variables -> Actions). Set API keys as secrets and
    things like the tournament id as variables.

    When using an .env file, the environment variables should be specified in the following format:
    METACULUS_TOKEN=1234567890

    The following environment variables are important:
    - METACULUS_TOKEN: your metaculus API token (go to https://www.metaculus.com/aib/ to get one)
    - OPENAI_API_KEY: your openai API key
    - ANTHROPIC_API_KEY: your anthropic API key
    - PERPLEXITY_API_KEY: your perplexity API key

    Running the bot
    ----------------------
    Run this bot using `poetry run python simple-forecast-bot.py`.
    By default, the bot will not submit any predictions. You need to change that
    setting int he code.
    """

    # define bot parameters
    use_perplexity = False
    submit_predictions = False
    metac_token = config("METACULUS_TOKEN")
    metac_base_url = "https://www.metaculus.com/api2"
    tournament_id = 3294
    llm_model_name = "gpt-4o"

    all_questions = []
    offset = 0

    # get all questions in the tournament and add them to the all_questions list
    while True:
        questions = list_questions(
            metac_base_url, metac_token, tournament_id, offset=offset
        )
        if len(questions) < 1:
            break # break the while loop if there are no more questions to process
        offset += len(questions) # update the offset for the next batch of questions
        all_questions.extend(questions)

    for question in all_questions:
        print("Forecasting ", question["id"], question["question"]["title"])

        # get news info from perplexity if the user wants to use it
        news_summary = call_perplexity(question["question"]["title"]) if use_perplexity else None

        prompt = build_prompt(
            question["question"]["title"],
            question["question"]["description"],
            question["question"].get("resolution_criteria", ""),
            question["question"].get("fine_print", ""),
            news_summary,
        )

        print(
            f"\n\n*****\nPrompt for question {question['id']}/{question['question']['title']}:\n{prompt} \n\n\n\n"
        )

        # get the language model to be used based on the name of the model
        llm_model = get_model(llm_model_name)
        # make a call to the language model
        response = llm_model.chat(
            messages=[ChatMessage(role=MessageRole.USER, content=prompt)]
        )
        llm_prediction = process_forecast_probabilty(response.message.content)
        rationale = response.message.content


        if llm_prediction is not None and submit_predictions:

            # post prediction
            post_url = f"{metac_base_url}/questions/{question['id']}/predict/"
            response = requests.post(
                post_url,
                json={"prediction": float(llm_prediction) / 100},
                headers={"Authorization": f"Token {metac_token}"},
            )
            response.raise_for_status()

            # post comment with rationale
            rationale = rationale + "\n\n" + "Used the following information from perplexity:\n\n" + news_summary
            comment_url = f"{metac_base_url}/comments/" # this is the url for the comments endpoint
            response = requests.post(
                comment_url,
                json={
                    "comment_text": rationale,
                    "submit_type": "N", # submit this as a private note
                    "include_latest_prediction": True,
                    "question": question["id"],
                },
                headers={"Authorization": f"Token {metac_token}"}, # your token is used to authenticate the request
            )

            print(f"Posted prediction for {question['id']}")

if __name__ == "__main__":
    main()