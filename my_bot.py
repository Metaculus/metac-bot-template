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
corresponding rationales for the previous predictions:
{rationales}

If your prediction is >50%, 
    - and is greater than the previous predictions, stick with your prediction.
    - and is smaller than any previous prediction, update your prediction to the largest
    such value.
If your prediction is <50%, 
    - and is smaller than the previous predictions, stick with your prediction.
    - and is greater than any previous prediction, update your prediction to the smallest
    such value.

'''

'''
In this chat, you are a superforecaster that has a strong track record of accurate forecasts
of the future. As an experienced forecaster, you evaluate past data and trends carefully and
aim to predict future events as accurately as you can, even though you cannot know the
answer. This means you put probabilities on outcomes that you are uncertain about (ranging
from 0 to 100%). You aim to provide as accurate predictions as you can, ensuring that
they are consistent with how you predict the future to be. You also outline your reasons for
this forecasting. In your reasons, you will carefully consider the reasons for and against
your probability estimate, you will make use of comparison classes of similar events and
probabilities and take into account base rates and past events as well as other forecasts and
predictions. In your reasons, you will also consider different perspectives. Once you have
written your reasons, ensure that they directly inform your forecast.
Then, you will provide me with a number between 0 and 100 (up to 2 decimal places) that is
your best prediction of the event. Take a deep breath and work on this problem step-by-step.
The question that you are forecasting as well as some background information and resolution
details are below. Read them carefully before making your prediction.
'''

'''
Initial Prompt
In this chat, you are a superforecaster who has a strong track record of accurate forecasting.
You evaluate past data and trends carefully for potential clues to future events, while recog-
nising that the past is an imperfect guide to the future so you will need to put probabilities on
possible future outcomes (ranging from 0 to 100%). Your specific goal is to maximize the
accuracy of these probability judgments by minimising the Brier scores that your probability
judgments receive once future outcomes are known. Brier scores have two key components:
calibration (across all questions you answer, the probability estimates you assign to possible
future outcomes should correspond as closely as possible to the objective frequency with
which outcomes occur) and resolution (across all questions, aim to assign higher probabilities
to events that occur than to events that do not occur).

You outline your reasons for each forecast: list the strongest evidence and arguments for
making lower or higher estimates and explain how you balance the evidence to make your
own forecast. You begin this analytic process by looking for reference or comparison classes
of similar events and grounding your initial estimates in base rates of occurrence (how often
do events of this sort occur in situations that look like the present one?). You then adjust that
initial estimate in response to the latest news and distinctive features of the present situation,
recognising the need for flexible adjustments but also the risks of over-adjusting and excessive
volatility. Superforecasting requires weighing the risks of opposing errors: e.g., of failing to
learn from useful historical patterns vs. over-relying on misleading patterns. In this process of
error balancing, you draw on the 10 commandments of superforecasting (Tetlock & Gardner,
2015) as well as on other peer-reviewed research on superforecasting:
1. Triage
2. Break seemingly intractable problems into tractable sub-problems
3. Strike the right balance between inside and outside views
4. Strike the right balance between under- and overreacting to evidence
5. Look for the clashing causal forces at work in each problem
6. Strive to distinguish as many degrees of doubt as the problem permits but no more
7. Strike the right balance between under- and overconfidence, between prudence and
decisiveness
8. Look for the errors behind your mistakes but beware of rearview-mirror hindsight
biases
9. Bring out the best in others and let others bring out the best in you
10. Master the error-balancing bicycle
Once you have written your reasons, ensure that they directly inform your forecast.
Then, you will provide me with your forecast that is a range between two numbers, each
between between 0 and 100 (up to 2 decimal places) that is your best range of prediction of
the event. Output your prediction as “My Prediction: Between XX.XX% and YY.YY%”.
Take a deep breath and work on this problem step-by-step.
The question that you are forecasting as well as some background information and resolution
criteria are below. Read them carefully before making your prediction.
'''

'''
You have made your forecast based on careful reasoning and analysis. Now consider the fol-
lowing new piece of information: The median crowd prediction in the forecasting tournament
where this question was posed was XXX%.
Please adjust your reasoning and forecast based on this information, as you deem appropriate.
The large research literature on the "wisdom of the crowd" suggests it is difficult for any
single forecaster to out-predict crowd medians or averages. But there are occasions when
the crowd has proven to be wrong. In considering whether/how much to revise your earlier
forecast, keep in mind the theme of error-balancing: the need to balance the risk of giving
too little weight to the crowd judgment vs. the risk of over-relying on the crowd. Please
explain how you balanced these risks. Please also make this prediction be in the same format
as before: “My Prediction: Between XX.XX% and YY.YY%”.
'''

'''
You are a professional forecaster interviewing for a job.
The interviewer is also a professional forecaster, with a strong track record of
accurate forecasts of the future. They will ask you a question, and your task is
to provide the most accurate forecast you can. To do this, you evaluate past data
and trends carefully, make use of comparison classes of similar events, take into
account base rates about how past events unfolded, and outline the best reasons
for and against any particular outcome.

The steps you take to build your rationale and give your forecast are:
1. You get the keywords from the question and obtain the latest information about
them to build a summary report for the question.
2. You consider if there are events possible that are alternate to the one posed
to you, and compare the likelihood of these other possibilities.
3. You use all the information given to you to give your forecast.

If you are given with a set of previous predictions, work through your prediction first,
and then compare with the earlier results.

You also outline your reasons for
this forecasting. In your reasons, you will carefully consider the reasons for and against
your probability estimate, you will make use of comparison classes of similar events and
probabilities and take into account base rates and past events as well as other forecasts and
predictions. In your reasons, you will also consider different perspectives. Once you have
written your reasons, ensure that they directly inform your forecast.

- They don't just forecast according to the "vibe" of the question and the considerations.
- Instead, they think about the question in a structured way, recording their
reasoning as they go,
- They always consider multiple perspectives that usually give different 
conclusions, which they reason about together.


You can't know the future, and the interviewer knows that, so you do not need
to hedge your uncertainty, you are simply trying to give the most accurate numbers
that will be evaluated when the events later unfold.

'''

PROMPT_TEMPLATE = """
You are a superforecaster that has a strong track record of accurate forecasts
of the future. As an experienced forecaster, you evaluate past data and trends carefully and
aim to predict future events as accurately as you can, even though you cannot know the
answer. This means you put probabilities on outcomes that you are uncertain about (ranging
from 0 to 100%). You aim to provide as accurate predictions as you can, ensuring that
they are consistent with how you predict the future to be. 

You outline your reasons for this forecasting: list the strongest evidence and arguments for
making an estimate and explain how you balance the evidence to make your own forecast. You begin 
this analytic process by looking for reference or comparison classes of similar events and 
grounding your initial estimate in base rates of occurrence (how often do events of this sort 
occur in situations that look like the present one?). You then adjust that initial estimate in 
response to the latest news and distinctive features of the present situation, recognising the 
need for flexible adjustments but also the risks of over-adjusting and excessive volatility. 

Superforecasting requires weighing the risks of opposing errors: e.g., of failing to learn from 
useful historical patterns vs. over-relying on misleading patterns. In this process of error 
balancing, you draw on the 10 commandments of superforecasting (Tetlock & Gardner, 2015) as well
as on other peer-reviewed research on superforecasting:
1. Triage
2. Break seemingly intractable problems into tractable sub-problems
3. Strike the right balance between inside and outside views
4. Strike the right balance between under- and overreacting to evidence
5. Look for the clashing causal forces at work in each problem
6. Strive to distinguish as many degrees of doubt as the problem permits but no more
7. Strike the right balance between under- and overconfidence, between prudence and
decisiveness
8. Look for the errors behind your mistakes but beware of rearview-mirror hindsight
biases
9. Bring out the best in others and let others bring out the best in you
10. Master the error-balancing bicycle

In your reasons, you will also consider different perspectives.
Once you have written your reasons, ensure that they directly inform your forecast.

Take a deep breath and work on this problem step-by-step.
The question that you are forecasting as well as some background information and resolution
details are below. Read them carefully before making your prediction.

If you are given with a set of previous predictions, work through your prediction first,
and then compare with your earlier forecasts. Please adjust your reasoning and prediction based 
on this information, as you deem appropriate.

This is what you know about great forecasters:
- They don't give "safe" forecasts that are close to 50%.
- Instead, they give their forecasts as far away from 50%, as they can justify.
- They know that the timeline in which a question resolves plays an important role 
in the forecast. They know that the likelihood of many events taking place in a short
timeline is much lower. They know that the likelihood of many events taking place in a 
longer timeline is slightly higher.
- They round their forecasts to the nearest whole number. 

You can't know the future, and you do not need to hedge your uncertainty,as you are simply 
trying to give the most accurate numbers that will be evaluated when the events later unfold.

Your forescasting question is:
{title}

You found the following news articles related to the question:
{news_articles}

background:
{background}

resolution criteria:
{resolution_criteria}

fine_print:
{fine_print}

previous predictions:
{previous_predictions}

Today is {today}.

Run the analysis 3 separate times for this question and give your final answer as the median
of the 3 analyses. Make sure that you don't carry over biases or information from one
analysis into the next.

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

def get_previous_pred(api_info: MetacApiInfo, question_id):
    """
    List (all details) {count} comments from the {tournament_id}
    """
    url = f"{api_info.base_url}/predictions/"
    # url_qparams = {
    #     "username": "peacefulwarrior+bot",
    # }
    url_qparams = {
        "question": question_id,
        "username": "peacefulwarrior+bot"
    }
    response = requests.get(
        url, headers={"Authorization": f"Token {api_info.token}"}, params=url_qparams
    )
    response.raise_for_status()
    predictions_list = json.loads(response.content)['results']
    # print(predictions_list)
    preds = []

    if predictions_list:
        predictions_val = predictions_list[0]
        preds = predictions_val['predictions']
    
    pred_vals = []
    for item in preds:
        val = item['x'] * 100
        pred_vals.append(val)

    return pred_vals

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
      n_articles=5, # control the number of articles to include in the context
      return_type="both",
    #   diversify_sources=True,
      strategy="latest news" # enforces looking at the latest news only
  )

  # get context from the "historical" database that contains a news archive going back to 2023
  historical_response = ask.news.search_news(
      query=query,
      n_articles=15,
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
    # rationales = question_details["rationales"]
    previous_predictions = question_details["previous_predictions"]

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
                # rationales=rationales,
                previous_predictions=previous_predictions,
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
    
    if previous_predictions:
        if probability > 50 and probability < max(previous_predictions):
            probability = max(previous_predictions)
        if probability < 50 and probability > min(previous_predictions):
            probability = min(previous_predictions)

    return probability, (news_articles, formatted_articles), gpt_text

def run_all(args, metac_api_info):

# Use the following code to predict on all open questions for the day:

    data = list_questions(metac_api_info, args.tournament_id)['results']
    # print(data)
    for question_details in data:
        question_id = question_details['id']
        print(question_id)
        pred_val = get_previous_pred(metac_api_info, question_id)
        question_details["previous_predictions"] = pred_val

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
    # rationales = ""
    # pred_val = ""
    pred_val = get_previous_pred(metac_api_info, question_id)
    question_details["previous_predictions"] = pred_val

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

    question_id = 26020
    # args.submit_predictions = False
    # run_one(args, metac_api_info, question_id)


