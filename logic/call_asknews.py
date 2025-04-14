import os
from typing import Dict

import asyncio
from asknews_sdk import AskNewsSDK
from autogen import ConversableAgent
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent

from logic.chat import validate_and_parse_response
from logic.utils import create_prompt
from utils.PROMPTS import HYDE_PROMPT
from utils.config import get_gpt_config

ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_SECRET = os.getenv("ASKNEWS_SECRET")
async def run_research(question: Dict[str,str]) -> str:
    research = ""
    if ASKNEWS_CLIENT_ID and ASKNEWS_SECRET:
        print("Running research...")
        try:
            research = await call_asknews(question)
        except:
            print("Error in research, retrying... in 60 seconds")
            await asyncio.sleep(60)
            research = await call_asknews(question)
    else:
        raise ValueError("No API key provided")

    print(f"########################\nResearch Found:\n{research}\n########################")

    return research


async def call_asknews(question_details:Dict[str,str]) -> str:
    """
    Use the AskNews `news` endpoint to get news context for your query.
    The full API reference can be found here: https://docs.asknews.app/en/reference#get-/v1/news/search
    """
    ask = AskNewsSDK(
        client_id=ASKNEWS_CLIENT_ID, client_secret=ASKNEWS_SECRET, scopes={"news"}
    )
    hyde_results = await hyde(question_details)

    # get the latest news related to the query (within the past 48 hours)
    hot_response = ask.news.search_news(
        query=hyde_results,  # your natural language query
        n_articles=5,  # control the number of articles to include in the context, originally 5
        return_type="both",
        strategy="latest news",  # enforces looking at the latest news only
    )

    # get context from the "historical" database that contains a news archive going back to 2023
    historical_response = ask.news.search_news(
        query=hyde_results,
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


async def hyde(question_details:Dict[str,str])->str:
    config = get_gpt_config(42, 0.7, "gpt-4o", 120)
    prompt = create_prompt(question_details, news="Empty...")
    agent = GPTAssistantAgent(name = "Hyde",instructions=HYDE_PROMPT,llm_config=config)
    hyde_reply = await agent.a_generate_reply(messages=[{"role": "user", "content": prompt}])
    result = validate_and_parse_response(hyde_reply['content'])
    return result.get("article",None)


