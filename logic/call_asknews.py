import asyncio
import os
from typing import Dict

from asknews_sdk import AskNewsSDK
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from logic.chat import validate_and_parse_response
from logic.utils import extract_question_details
from utils.PROMPTS import HYDE_PROMPT

ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_SECRET = os.getenv("ASKNEWS_SECRET")


async def run_research(question: Dict[str, str], use_hyde: bool = True) -> str:
    research = ""
    if ASKNEWS_CLIENT_ID and ASKNEWS_SECRET:
        print("Running research...")
        try:
            research = await call_asknews(question, use_hyde=use_hyde)
        except:
            print("Error in research, retrying... in 60 seconds")
            await asyncio.sleep(60)
            research = await call_asknews(question, use_hyde=use_hyde)
    else:
        raise ValueError("No API key provided")

    print(f"########################\nResearch Found:\n{research}\n########################")

    return research


async def call_asknews(question_details: Dict[str, str], use_hyde: bool = True) -> str:
    """
    Use the AskNews `news` endpoint to get news context for your query.
    The full API reference can be found here: https://docs.asknews.app/en/reference#get-/v1/news/search
    """
    ask = AskNewsSDK(
        client_id=ASKNEWS_CLIENT_ID, client_secret=ASKNEWS_SECRET, scopes={"news"}
    )
    if use_hyde:
        query = await hyde(question_details)
    else:
        query = asknews_query_builder(question_details)

    # get the latest news related to the query (within the past 48 hours)
    hot_response = ask.news.search_news(
        query=query,  # your natural language query
        n_articles=5,  # control the number of articles to include in the context, originally 5
        return_type="both",
        strategy="latest news",  # enforces looking at the latest news only
    )

    # get context from the "historical" database that contains a news archive going back to 2023
    historical_response = ask.news.search_news(
        query=query,
        n_articles=10,
        return_type="both",
        strategy="news knowledge",  # looks for relevant news within the past 60 days
    )

    hot_articles = hot_response.as_dicts
    historical_articles = historical_response.as_dicts
    formatted_articles = "Here are the relevant news articles:\n\n"

    if hot_articles:
        hot_articles = [article.__dict__ for article in hot_articles]
        hot_articles = sorted(hot_articles, key=lambda x: x["pub_date"], reverse=True)

        for article in hot_articles:
            pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
            formatted_articles += f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {pub_date}\nSource:[{article['source_id']}]({article['article_url']})\n\n"

    if historical_articles:
        historical_articles = [article.__dict__ for article in historical_articles]
        historical_articles = sorted(
            historical_articles, key=lambda x: x["pub_date"], reverse=True
        )

        for article in historical_articles:
            pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
            formatted_articles += f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {pub_date}\nSource:[{article['source_id']}]({article['article_url']})\n\n"

    if not hot_articles and not historical_articles:
        formatted_articles += "No articles were found.\n\n"
        return formatted_articles

    return formatted_articles

def asknews_query_builder(question_details: Dict[str, str]) -> str:
    """
    Build a query for AskNews based on the question details.
    """
    title, description, _, _, _, _= extract_question_details(question_details)
    query = f"{title} {description}"
    return query



async def hyde(question_details: Dict[str, str]) -> str:
    model_client = OpenAIChatCompletionClient(model="gpt-4.1", temperature=1)
    title, description, fine_print, resolution_criteria, forecast_date, aggregations = extract_question_details(question_details)
    full_prompt = (
        f"##Forecast Date: {forecast_date}\n\n##Question:\n{title}\n\n##Description:\n{description}\n\n##Fine Print:\n"
        f"{fine_print}\n\n##Resolution Criteria:\n{resolution_criteria}")
    agent = AssistantAgent(name="Hyde", system_message=HYDE_PROMPT, model_client=model_client)
    hyde_reply = await agent.run(task=full_prompt)
    result = validate_and_parse_response(hyde_reply.messages[1].content)
    return result.get("article", None)
