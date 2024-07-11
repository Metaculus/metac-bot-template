#!/usr/bin/env python

import json

from attr import dataclass
import requests
from decouple import config
import datetime
import re
from openai import OpenAI
from asknews_sdk import AskNewsSDK
import argparse

@dataclass
class MetacApiInfo:
    token: str
    base_url: str


''' 

You know that great forecasters don't
just forecast according to the "vibe" of the question and the considerations.
Instead, they think about the question in a structured way, recording their
reasoning as they go, and they always consider multiple perspectives that
usually give different conclusions, which they reason about together.

You know that rookie forecasters sometimes tend to give "safe" forecasts that are 
close to 50%. You also know that great forecasters instead, give their forecasts 
as far away from 50%, as they can justify.
You also know that great forecasters round their forecasts to the nearest whole
number.

Remember that not everything your research assistant says will be accurate. Always
trust the news articles over the summary of your research assistant.

Your research assistant says:
{summary_report}

'''

PROMPT_TEMPLATE = """
You are a professional forecaster interviewing for a job.
The interviewer is also a professional forecaster, with a strong track record of
accurate forecasts of the future. They will ask you a question, and your task is
to provide the most accurate forecast you can. To do this, you evaluate past data
and trends carefully, make use of comparison classes of similar events, take into
account base rates about how past events unfolded, and outline the best reasons
for and against any particular outcome.

This is what you know about great forecasters:
- They don't just forecast according to the "vibe" of the question and the considerations.
- Instead, they think about the question in a structured way, recording their
reasoning as they go,
- They always consider multiple perspectives that usually give different 
conclusions, which they reason about together.
- They don't give "safe" forecasts that are close to 50%.
- Instead, they give their forecasts as far away from 50%, as they can justify.
- They know that the timeline in which a question resolves plays an important role 
in the forecast. They know that the likelihood of many events taking place in a short
timeline is much lower. They know that the likelihood of many events taking place in a 
longer timeline is slightly higher.
- They round their forecasts to the nearest whole number. 

You can't know the future, and the interviewer knows that, so you do not need
to hedge your uncertainty, you are simply trying to give the most accurate numbers
that will be evaluated when the events later unfold.

The steps you take to build your rationale and give your forecast are:
1. You get the keywords from the question and obtain the latest information about
them to build a summary report for the question.
2. You consider if there are events possible that are alternate to the one posed
to you, and compare the likelihood of these other possibilities.
3. You use all the information given to you to give your forecast.

Your interview question is:
{title}

You found the following news articles related to the question:
{news_articles}

background:
{background}

resolution criteria:
{resolution_criteria}

fine_print:
{fine_print}

Today is {today}.

You write your rationale and give your final answer as: "Probability: ZZ%", 0-100
"""

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


def list_questions(api_info: MetacApiInfo, tournament_id: int, offset=0, count=15):
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
    print(f"Token {api_info.token}")
    response = requests.get(
        url, headers={"Authorization": f"Token {api_info.token}"}, params=url_qparams
    )
    response.raise_for_status()
    return json.loads(response.content)

def get_asknews_context(query):
  """
  Use the AskNews `news` endpoint to get news context for your query.
  The full API reference can be found here: https://docs.asknews.app/en/reference#get-/v1/news/search
  """
  asknews_client_id = config("ASKNEWS_CLIENT_ID", default="-", cast=str)
  asknews_secret = config("ASKNEWS_SECRET", default="-", cast=str)
  ask = AskNewsSDK(
      client_id=asknews_client_id,
      client_secret=asknews_secret,
      scopes=["news"]
  )

  # # get the latest news related to the query (within the past 48 hours)
  hot_response = ask.news.search_news(
      query=query, # your natural language query
      n_articles=10, # control the number of articles to include in the context
      return_type="both",
      diversify_sources=True,
      strategy="latest news" # enforces looking at the latest news only
  )

  # get context from the "historical" database that contains a news archive going back to 2023
  historical_response = ask.news.search_news(
      query=query,
      n_articles=20,
      return_type="both",
      diversify_sources=True,
      historical=True,
    #   strategy="news knowledge" # looks for relevant news within the past 60 days
  )

  # you can also specify a time range for your historical search if you want to
  # slice your search up periodically.
  # now = datetime.datetime.now().timestamp()
  # start = (datetime.datetime.now() - datetime.timedelta(days=100)).timestamp()
  # historical_response = ask.news.search_news(
  #     query=query,
  #     n_articles=20,
  #     return_type="both",
  #     historical=True,
  #     start_timestamp=int(start),
  #     end_timestamp=int(now)
  # )

  llm_context = hot_response.as_string + historical_response.as_string
  formatted_articles = format_asknews_context(
      hot_response.as_dicts, historical_response.as_dicts)
  return llm_context, formatted_articles

def format_asknews_context(hot_articles, historical_articles):
  """
  Format the articles for posting to Metaculus.
  """

  formatted_articles = "Here are the relevant news articles:\n\n"

  if hot_articles:
    hot_articles = [article.__dict__ for article in hot_articles]
    hot_articles = sorted(
        hot_articles, key=lambda x: x['pub_date'], reverse=True)

    for article in hot_articles:
        pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
        formatted_articles += f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {pub_date}\nSource:[{article['source_id']}]({article['article_url']})\n\n"

  if historical_articles:
    historical_articles = [article.__dict__ for article in historical_articles]
    historical_articles = sorted(
        historical_articles, key=lambda x: x['pub_date'], reverse=True)

    for article in historical_articles:
        pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
        formatted_articles += f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {pub_date}\nSource:[{article['source_id']}]({article['article_url']})\n\n"

  if not hot_articles and not historical_articles:
    formatted_articles += "No articles were found.\n\n"
    return formatted_articles

  formatted_articles += f"*Generated by AI at [AskNews](https://asknews.app), check out the [API](https://docs.asknews.app) for more information*."

  return formatted_articles

def call_perplexity(query):
    url = "https://api.perplexity.ai/chat/completions"
    api_key = config("PERPLEXITY_API_KEY", default="-", cast=str)
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    payload = {
        "model": "llama-3-sonar-large-32k-online",
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

def get_gpt_prediction(question_details):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    client = OpenAI(api_key=config("OPENAI_API_KEY", default="", cast=str))

    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]

    news_articles = ""
    # summary_report = 0

    # Comment this line to not use AskNews
    news_articles, formatted_articles = get_asknews_context(title)

    # Comment this line to not use perplexity
    # summary_report = call_perplexity(title)

    chat_completion = client.chat.completions.create(
        model="gpt-4o",
        # model="gpt-3.5-turbo-16k",
        # model="gpt-3.5-turbo",
        messages=[
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(
                title=title,
                # summary_report=summary_report,
                news_articles=news_articles,
                today=today,
                background=background,
                resolution_criteria=resolution_criteria,
                fine_print=fine_print,
            )
        }
        ]
    )

    gpt_text = chat_completion.choices[0].message.content

    # Regular expression to find the number following 'Probability: '
    probability_match = find_number_before_percent(gpt_text)

    # Extract the number if a match is found
    probability = None
    if probability_match:
        probability = int(probability_match) # int(match.group(1))
        print(f"The extracted probability is: {probability}%")
        probability = min(max(probability, 1), 99) # To prevent extreme forecasts

    return probability, (news_articles, formatted_articles), gpt_text

def run_all(args, metac_api_info):

# Use the following code to predict on all open questions for the day:

    data = list_questions(metac_api_info, args.tournament_id)['results']
    # print(data)
    for question_details in data:
        question_id = question_details['id']
        print(question_id)

        prediction, asknews_result, gpt_result = get_gpt_prediction(question_details)
        print("GPT predicted: ", prediction, asknews_result, gpt_result)

        if prediction is not None and args.submit_predictions:
            post_question_prediction(metac_api_info, question_id, prediction)
            comment = "\n\nAskNews sources\n\n" + asknews_result[1] + "\n\n#########\n\n" + "\n\n#########\n\n" + "GPT\n\n" + gpt_result
            post_question_comment(metac_api_info, question_id, comment)

def run_one(args, metac_api_info, question_id):

# Use the following code to predict on one question:
    question_details = get_question_details(metac_api_info, question_id)
    # print(question_details)

    prediction, asknews_result, gpt_result = get_gpt_prediction(question_details)
    print("GPT predicted: ", prediction, asknews_result, gpt_result)

    if prediction is not None and args.submit_predictions:
        post_question_prediction(metac_api_info, question_id, prediction)
        comment = "\n\nAskNews sources\n\n" + asknews_result[1] + "\n\n#########\n\n" + "\n\n#########\n\n" + "GPT\n\n" + gpt_result
        post_question_comment(metac_api_info, question_id, comment)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="A simple forecasting bot based on LLMs"
    )
    parser.add_argument(
        "--submit_predictions",
        help="Submit the predictions to Metaculus",
        default=True,
        action="store_true",
    )
    parser.add_argument(
        "--use_perplexity",
        help="Use perplexity.ai to search some up to date info about the forecasted question",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--use_asknews",
        help="Use asknews.ai to search some up to date news articles about the forecasted question",
        default=True,
        action="store_true",
    )
    parser.add_argument(
        "--number_forecasts",
        type=int,
        default=1,
        help="The number of LLM forecasts to average per question",
    )
    parser.add_argument(
        "--tournament_id",
        type=int,
        help="The tournament ID where to predict",
        default=config("TOURNAMENT_ID", default=0, cast=int),
    )

    args = parser.parse_args()

    metac_api_info = MetacApiInfo(
        token=config("METACULUS_TOKEN", default="-"),
        base_url="https://www.metaculus.com/api2",
    )

    # asyncio.run(main())
    run_all(args, metac_api_info)

    question_id = 25767
    # args.submit_predictions = False
    # run_one(args, metac_api_info, question_id)


