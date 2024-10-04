#!/usr/bin/env python

import asyncio
import json
import logging
import statistics

from attr import dataclass
import requests
from decouple import config
import datetime
import re
from jinja2 import Template

from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.ollama import Ollama
from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings

import argparse


@dataclass
class MetacApiInfo:
    token: str
    base_url: str


PROMPT_TEMPLATE = """
You are a professional forecaster interviewing for a job.

Your interview question is:
{{title}}

background:
{{description}}

{{resolution_criteria}}

{{fine_print}}


{% if summary_report %}
Your research assistant says:
{{summary_report}}
{% endif %}


Today is {{today}}.

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) What the outcome would be if nothing changed.
(c) What you would forecast if there was only a quarter of the time left.
(d) What you would forecast if there was 4x the time left.

You write your rationale and then the last thing you write is your final answer as: "Probability: ZZ%", 0-100
"""


def build_prompt(question_details, summary_report=None):
    prompt_jinja = Template(PROMPT_TEMPLATE)
    params = {
        "today": datetime.datetime.now().strftime("%Y-%m-%d"),
        "summary_report": summary_report,
        **question_details,
    }
    return prompt_jinja.render(params)


def clamp(x, a, b):
    return min(b, max(a, x))


def find_number_before_percent(s):
    # Use a regular expression to find all numbers followed by a '%'
    matches = re.findall(r"(\d+)%", s)
    if matches:
        # Return the last number found before a '%'
        return clamp(int(matches[-1]), 1, 99)
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
    if not response.ok:
        logging.error(
            f"Failed posting a comment on question {question_id}: {response.text}"
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
    response.raise_for_status()
    if not response.ok:
        logging.error(
            f"Failed posting a prediction on question {question_id}: {response.text}"
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
        "format": "json",
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
        "model": "llama-3.1-sonar-large-128k-chat",
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
    print(
        f"\n\nCalled perplexity with:\n----\n{json.dumps(payload)}\n---\n, and got\n:",
        content,
    )
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
        case "anthropic":
            tokenizer = Anthropic().tokenizer
            Settings.tokenizer = tokenizer
            return Anthropic(
                api_key=config("ANTHROPIC_API_KEY", default=""),
                model="claude-3-5-sonnet-20240620",
            )
        case "o1-preview":
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
        choices=["gpt-4o", "gpt-3.5-turbo", "anthropic", "o1-preview"],
        default="gpt-4o",
        help="The model to use, one of the options listed",
    )
    parser.add_argument(
        "--metac_base_url",
        type=str,
        help="The base URL for the metaculus API",
        default=config("API_BASE_URL", default="https://beta.metaculus.com/api2", cast=str),
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

    if args.number_forecasts < 1:
        print("number_forecasts must be larger than 0")
        return

    offset = 0
    while True:
        questions = list_questions(
            metac_api_info, args.tournament_id, offset=offset, count=5
        )
        print("Handling questions: ", [q["id"] for q in questions])
        if len(questions) < 1:
            break

        offset += len(questions)

        pp_questions = [
            (
                question,
                call_perplexity(question["question"]["title"]) if args.use_perplexity else None,
            )
            for question in questions
        ]
        prompts = [
            build_prompt(
                {
                    "title": question["question"]["title"],
                    "description": question["question"]["description"],
                    "resolution_criteria": question["question"].get("resolution_criteria", ""),
                    "fine_print": question["question"].get("fine_print", ""),
                },
                pp_result,
            )
            for question, pp_result in pp_questions
        ]

        for question, prompt in zip(questions, prompts):
            print(
                f"\n\n*****\nPrompt for question {question['id']}/{question['question']['title']}:\n{prompt} \n\n\n\n"
            )

        all_predictions = {q["id"]: [] for q in questions}
        for round in range(args.number_forecasts):
            results = await asyncio.gather(
                *[llm_predict_once(llm_model, prompt) for prompt in prompts],
            )

            for (prediction, reasoning), question in zip(results, questions):
                id = question["id"]
                print(
                    f"\n\n****\n(round {round})Forecast for {id}: {prediction}, Rationale:\n {reasoning}"
                )
                if prediction is not None:
                    post_question_prediction(metac_api_info, id, float(prediction))
                    post_question_comment(metac_api_info, id, reasoning)
                    all_predictions[id].append(float(prediction))

        if args.number_forecasts > 1:
            for q in questions:
                id = q["id"]
                q_predictions = all_predictions[id]
                if len(q_predictions) < 1:
                    continue
                median = statistics.median(q_predictions)
                post_question_prediction(metac_api_info, id, median)
                post_question_comment(
                    metac_api_info,
                    id,
                    f"Computed the median of the last {len(q_predictions)} predictions: {median}",
                )

        for question, perplexity_result in pp_questions:
            id = question["id"]
            post_question_comment(
                metac_api_info,
                id,
                f"##Used perplexity info:\n\n {perplexity_result}",
            )


if __name__ == "__main__":
    asyncio.run(main())
