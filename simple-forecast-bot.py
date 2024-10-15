#!/usr/bin/env python

import json

import requests
from decouple import config
import datetime
import re


def main():
    """
    Main function to run the forecasting bot. This function accesses the questions
    for a given tournament, fetches information about them, and then uses an LLM
    to generate a forecast.

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
    - PERPLEXITY_API_KEY: your perplexity API key, if you are using perplexity (i.e. use_perplexity is set to Tru )

    Running the bot
    ----------------------
    Run this bot using `poetry run python simple-forecast-bot.py`.
    By default, the bot will not submit any predictions. You need set the "submit_predictions" flag to True in the code.
    """

    # define bot parameters
    use_perplexity = True
    submit_predictions = True
    metac_token = config("METACULUS_TOKEN")
    metac_base_url = "https://www.metaculus.com/api2"
    tournament_id = 32506
    llm_model_name = config("LLM_MODEL_NAME", default="gpt-4o")

    all_questions = []
    offset = 0

    # get all questions in the tournament and add them to the all_questions list
    while True:
        questions = list_questions(
            metac_base_url, metac_token, tournament_id, offset=offset
        )
        if len(questions) < 1:
            break  # break the while loop if there are no more questions to process
        offset += len(questions)  # update the offset for the next batch of questions
        all_questions.extend(questions)

    for question in all_questions:
        print("Forecasting ", question["id"], question["question"]["title"])

        # get news info from perplexity if the user wants to use it
        news_summary = (
            call_perplexity(question["question"]["title"]) if use_perplexity else None
        )

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

        llm_output = call_llm_model(llm_model_name, prompt)

        llm_prediction = process_forecast_probabilty(llm_output)
        print(f"\n\n*****\nLLM prediction: {llm_prediction}\n*****\n")
        rationale = llm_output

        if llm_prediction is not None and submit_predictions:
            # post prediction
            post_url = f"{metac_base_url}/questions/{question['id']}/predict/"
            response = requests.post(
                post_url,
                json={"prediction": float(llm_prediction) / 100},
                headers={"Authorization": f"Token {metac_token}"},
            )

            if not response.ok:
                raise Exception(response.text)

            # post comment with rationale
            rationale = (
                rationale
                + "\n\n"
                + "Used the following information from perplexity:\n\n"
                + news_summary
            )
            comment_url = f"{metac_base_url}/comments/"  # this is the url for the comments endpoint
            response = requests.post(
                comment_url,
                json={
                    "comment_text": rationale,
                    "submit_type": "N",  # submit this as a private note
                    "include_latest_prediction": True,
                    "question": question["id"],
                },
                headers={
                    "Authorization": f"Token {metac_token}"
                },  # your token is used to authenticate the request
            )

            print(f"\n\n*****\nPosted prediction for {question['id']}\n*****\n")


def build_prompt(
    title: str,
    description: str,
    resolution_criteria: str,
    fine_print: str,
    news_info: str | None = None,
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
        number = min(99, max(1, number))  # clamp the number between 1 and 99
        return number
    else:
        return None


def list_questions(
    base_url: str, metac_token: str, tournament_id: int, offset=0, count=10
):
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
        "limit": count,  # the number of questions to return
        "offset": offset,  # pagination offset
        "has_group": "false",
        "order_by": "-activity",  # order by activity (most recent questions first)
        "forecast_type": "binary",  # only binary questions are returned
        "project": tournament_id,  # only questions in the specified tournament are returned
        "status": "open",  # only open questions are returned
        "format": "json",  # return results in json format
        "type": "forecast",  # only forecast questions are returned
        "include_description": "true",  # include the description in the results
    }
    url = f"{base_url}/questions/"  # url for the questions endpoint
    response = requests.get(
        url, headers={"Authorization": f"Token {metac_token}"}, params=url_qparams
    )
    # you can verify what this is doing by looking at
    # https://www.metaculus.com/api2/questions/?format=json&has_group=false&limit=5&offset=0&order_by=-activity&project=3294&status=open&type=forecast
    # in the browser. The URL works as follows:
    # base_url/questions/, then a "?"" before the first url param and then a "&"
    # between additional parameters

    if not response.ok:
        raise Exception(response.text)

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
                "role": "system",  # this is a system prompt designed to guide the perplexity assistant
                "content": """
You are an assistant to a superforecaster.
The superforecaster will give you a question they intend to forecast on.
To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
You do not produce forecasts yourself.
""",
            },
            {
                "role": "user",  # this is the actual prompt we ask the perplexity assistant to answer
                "content": query,
            },
        ],
    }
    response = requests.post(url=url, json=payload, headers=headers)

    if not response.ok:
        raise Exception(response.text)

    content = response.json()["choices"][0]["message"]["content"]
    print(
        f"\n\nCalled perplexity with:\n----\n{json.dumps(payload)}\n---\n, and got\n:",
        content,
    )
    return content


def call_llm_model(model_name: str, prompt: str):
    extra_headers = {}
    extra_params = {}
    proxy_base_url = "https://old.metaculus.com"
    #proxy_base_url = "http://localhost:3000"

    def get_content_openai(x):
        return x["choices"][0]["message"]["content"]

    def get_content_anthropic(x):
        return x["content"][0]["text"]

    if model_name in ["gpt-4o", "gpt-3.5-turbo"]:
        url = f"{proxy_base_url}/proxy/openai/v1/chat/completions/"
        get_content = get_content_openai
    elif model_name == "claude-3-5-sonnet-20240620":
        extra_headers = {"anthropic-version": "2023-06-01"}
        # modify this as you see fit
        extra_params = {"max_tokens": 4096}
        url = f"{proxy_base_url}/proxy/anthropic/v1/messages/"
        get_content = get_content_anthropic
    else:
        raise ValueError(f"Model {model_name} not supported")

    headers = {
        "Authorization": f"Token {config('METACULUS_TOKEN')}",
        "Content-Type": "application/json",
        **extra_headers,
    }

    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        **extra_params,
    }

    response = requests.post(url, headers=headers, json=data)
    if not response.ok:
        raise Exception(response.text)

    return get_content(response.json())


if __name__ == "__main__":
    main()
