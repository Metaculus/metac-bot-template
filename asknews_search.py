from asknews_sdk import AskNewsSDK
from config import ASKNEWS_CLIENT_ID, ASKNEWS_SECRET


def call_asknews_hot(question: str) -> str:
    """
    Get the latest news related to the query (within the past 48 hours).
    """
    if not ASKNEWS_CLIENT_ID or not ASKNEWS_SECRET:
        return "AskNews credentials not provided."
    
    ask = AskNewsSDK(
        client_id=ASKNEWS_CLIENT_ID, client_secret=ASKNEWS_SECRET, scopes=set(["news"])
    )

    try:
        hot_response = ask.news.search_news(
            query=question,
            n_articles=6,
            return_type="both",
            strategy="latest news",
        )

        hot_articles = hot_response.as_dicts
        formatted_articles = "Here are the latest news articles:\n\n"

        if hot_articles:
            hot_articles = [article.__dict__ for article in hot_articles]
            hot_articles = sorted(hot_articles, key=lambda x: x["pub_date"], reverse=True)

            for article in hot_articles:
                pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
                formatted_articles += f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {pub_date}\nSource:[{article['source_id']}]({article['article_url']})\n\n"
        else:
            formatted_articles += "No latest articles were found.\n\n"

        return formatted_articles
        
    except Exception as e:
        return f"Error calling AskNews hot API: {str(e)}"


def call_asknews_historical(question: str) -> str:
    """
    Get context from the historical database (news archive going back to 2023).
    """
    if not ASKNEWS_CLIENT_ID or not ASKNEWS_SECRET:
        return "AskNews credentials not provided."
    
    ask = AskNewsSDK(
        client_id=ASKNEWS_CLIENT_ID, client_secret=ASKNEWS_SECRET, scopes=set(["news"])
    )

    try:
        historical_response = ask.news.search_news(
            query=question,
            n_articles=10,
            return_type="both",
            strategy="news knowledge",
        )

        historical_articles = historical_response.as_dicts
        formatted_articles = "Here are the relevant historical news articles:\n\n"

        if historical_articles:
            historical_articles = [article.__dict__ for article in historical_articles]
            historical_articles = sorted(
                historical_articles, key=lambda x: x["pub_date"], reverse=True
            )

            for article in historical_articles:
                pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
                formatted_articles += f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {pub_date}\nSource:[{article['source_id']}]({article['article_url']})\n\n"
        else:
            formatted_articles += "No historical articles were found.\n\n"

        return formatted_articles
        
    except Exception as e:
        return f"Error calling AskNews historical API: {str(e)}"


def call_asknews(question: str) -> str:
    """
    Use the AskNews `news` endpoint to get news context for your query.
    Sequential calls to avoid concurrent API usage.
    The full API reference can be found here: https://docs.asknews.app/en/reference#get-/v1/news/search
    """
    if not ASKNEWS_CLIENT_ID or not ASKNEWS_SECRET:
        return "AskNews credentials not provided."
    
    # Call hot articles first, then historical to avoid concurrent API calls
    hot_articles = call_asknews_hot(question)
    historical_articles = call_asknews_historical(question)
    return hot_articles + "\n" + historical_articles