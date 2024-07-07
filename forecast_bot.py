#!/usr/bin/env python

import asyncio
import json

from attr import dataclass
import requests
from decouple import config
import datetime
import re
from jinja2 import Template

from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.ollama import Ollama
import argparse


@dataclass
class MetacApiInfo:
    token: str
    base_url: str


PROMPT_TEMPLATE = """
You are a professional forecaster interviewing for a job.
The interviewer is also a professional forecaster, with a strong track record of
accurate forecasts of the future. They will ask you a question, and your task is
to provide the most accurate forecast you can. To do this, you evaluate past data
and trends carefully, make use of comparison classes of similar events, take into
account base rates about how past events unfolded, and outline the best reasons
for and against any particular outcome. You know that great forecasters don't
just forecast according to the "vibe" of the question and the considerations.
Instead, they think about the question in a structured way, recording their
reasoning as they go, and they always consider multiple perspectives that
usually give different conclusions, which they reason about together.
You can't know the future, and the interviewer knows that, so you do not need
to hedge your uncertainty, you are simply trying to give the most accurate numbers
that will be evaluated when the events later unfold.

Your interview question is:
{{title}}

{% if summary_report %}
Your research assistant says:
{{summary_report}}
{% endif %}

background:
{{description}}

fine_print:
{{fine_print}}

Today is {{today}}.

You write your rationale and give your final answer as: "Probability: ZZ%", 0-100
"""


def build_prompt(question_details, summary_report=None):
    prompt_jinja = Template(PROMPT_TEMPLATE)
    params = {
        "today": datetime.datetime.now().strftime("%Y-%m-%d"),
        "summary_report": summary_report,
        **question_details,
    }
    return prompt_jinja.render(params)


def find_number_before_percent(s):
    # Use a regular expression to find all numbers followed by a '%'
    matches = re.findall(r"(\d+)%", s)
    if matches:
        # Return the last number found before a '%'
        return int(matches[-1])
    else:
        # Return None if no number found
        return None


def post_question_comment(api_info: MetacApiInfo, question_id: int, comment_text: str):
    """
    Post a comment on the question page as the bot user.
    """

    response = requests.post(
        f"{api_info.base_url}/comments/",
        json={
            "comment_text": comment_text,
            "submit_type": "N",
            "include_latest_prediction": True,
            "question": question_id,
        },
        headers={"Authorization": f"Token {api_info.token}"},
    )
    return response.json, response.ok


def post_question_prediction(
    api_info: MetacApiInfo, question_id: int, prediction_percentage: float
):
    """
    Post a prediction value (between 1 and 100) on the question.
    """
    url = f"{api_info.base_url}/questions/{question_id}/predict/"
    response = requests.post(
        url,
        json={"prediction": float(prediction_percentage) / 100},
        headers={"Authorization": f"Token {api_info.token}"},
    )
    return response.json, response.ok


def get_question_details(api_info: MetacApiInfo, question_id):
    """
    Get all details about a specific question.
    """
    url = f"{api_info.base_url}/questions/{question_id}/"
    response = requests.get(
        url,
        headers={"Authorization": f"Token {api_info.token}"},
    )
    response.raise_for_status()
    return json.loads(response.content)


def list_questions(api_info: MetacApiInfo, tournament_id: int, offset=0, count=10):
    """
    List (all details) {count} questions from the {tournament_id}
    """
    url_qparams = {
        "limit": count,
        "offset": offset,
        "has_group": "false",
        "order_by": "-activity",
        "forecast_type": "binary",
        "project": tournament_id,
        "status": "open",
        "type": "forecast",
        "include_description": "true",
    }
    url = f"{api_info.base_url}/questions/"
    response = requests.get(
        url, headers={"Authorization": f"Token {api_info.token}"}, params=url_qparams
    )
    response.raise_for_status()
    data = json.loads(response.content)
    return data["results"]


def call_perplexity(query):
    url = "https://api.perplexity.ai/chat/completions"
    api_key = config("PERPLEXITY_API_KEY", default="-")
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    payload = {
        "model": "llama-3-sonar-large-32k-chat",
        "messages": [
            {
                "role": "system",
                "content": """
You are an assistant to a superforecaster.
The superforecaster will give you a question they intend to forecast on.
To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
You do not produce forecasts yourself.
""",
            },
            {"role": "user", "content": query},
        ],
    }
    response = requests.post(url=url, json=payload, headers=headers)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return content


def get_model(model_name: str):
    match model_name:
        case "gpt-4o":
            return OpenAI(
                api_key=config("OPENAI_API_KEY", default=""), model=model_name
            )
        case "gpt-3.5-turbo":
            return OpenAI(
                api_key=config("OPENAI_API_KEY", default=""), model=model_name
            )
    return None


async def llm_predict_once(chat_model, prompt):

    response = await chat_model.achat(
        messages=[ChatMessage(role=MessageRole.USER, content=prompt)]
    )

    probability_match = find_number_before_percent(response.message.content)
    return probability_match, response.message.content


async def main():
    parser = argparse.ArgumentParser(
        description="A simple forecasting bot based on LLMs"
    )
    parser.add_argument(
        "--submit_predictions",
        help="Submit the predictions to Metaculus",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--use_perplexity",
        help="Use perplexity.ai to search some up to date info about the forecasted question",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--number_forecasts",
        type=int,
        default=1,
        help="The number of LLM forecasts to average per question",
    )
    parser.add_argument(
        "--metac_token_env_name",
        type=str,
        help="The name of the env variable where to read the metaculus toekn from",
        default="METACULUS_TOKEN",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        choices=["gpt-4o", "gpt-3.5", "ollama:llama3"],
        default="gpt-4o",
        help="The model to use, one of the options listed",
    )
    parser.add_argument(
        "--metac_base_url",
        type=str,
        help="The base URL for the metaculus API",
        default=config("API_BASE_URL", default="https://www.metaculus.com/api2", cast=str),
    )
    parser.add_argument(
        "--tournament_id",
        type=int,
        help="The tournament ID where to predict",
        default=config("TOURNAMENT_ID", default=0, cast=int),
    )

    args = parser.parse_args()

    metac_api_info = MetacApiInfo(
        token=config(args.metac_token_env_name, default="-"),
        base_url=args.metac_base_url,
    )

    llm_model = get_model(args.llm_model)

    offset = 0
    while True:
        questions = list_questions(
            metac_api_info, args.tournament_id, offset=offset, count=5
        )
        if len(questions) < 1:
            break

        offset += len(questions)

        print("Handling questions: ", [q["id"] for q in questions])

        prompts = [
            build_prompt(
                question,
                call_perplexity(question["title"]) if args.use_perplexity else None,
            )
            for question in questions
        ]

        results = await asyncio.gather(
            *[llm_predict_once(llm_model, prompt) for prompt in prompts],
        )

        for (prediction, reasoning), question in zip(results, questions):
            print(f"Question id {question['id']} prediction: {prediction}")
            post_question_prediction(metac_api_info, question["id"], float(prediction))
            post_question_comment(metac_api_info, question["id"], reasoning)


if __name__ == "__main__":
    asyncio.run(main())
