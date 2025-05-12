import os
import requests

def get_related_markets_from_adjacent_news(question: str) -> str:
    """
    Given a question string, use the Adjacent News API to find related markets and return them as a formatted string.
    Only include markets with volume >= 1000, sorted by volume descending.
    """
    api_key = os.getenv("ADJACENT_NEWS_API_KEY")
    if not api_key:
        raise ValueError("ADJACENT_NEWS_API_KEY not set in environment.")
    base_url = "https://api.data.adj.news/api/search/query"
    params = {"q": question}
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.request("GET", base_url, params=params, headers=headers)
    print(f"DEBUG: Status code: {response.status_code}")
    print(f"DEBUG: Response text: {response.text}")
    if not response.ok:
        raise RuntimeError(f"Adjacent News API error: {response.text}")
    data = response.json()
    print(f"DEBUG: Parsed data: {data}")
    if not data or "data" not in data or not data["data"]:
        return "No related markets found."
    filtered_markets = []
    for market in data["data"]:
        volume = market.get("volume", "N/A")
        try:
            vol_num = float(volume)
        except (ValueError, TypeError):
            vol_num = 0
        if vol_num >= 1000:
            market["_parsed_volume"] = vol_num  # Store parsed volume for sorting
            filtered_markets.append(market)
    if not filtered_markets:
        return "No related markets found with volume >= 1000."
    # Sort by volume descending
    filtered_markets.sort(key=lambda m: m["_parsed_volume"], reverse=True)
    formatted = "Related Markets from Adjacent News (volume >= 1000, sorted by volume):\n"
    for market in filtered_markets:
        name = market.get("name", market.get("question", "Unnamed Market"))
        platform = market.get("platform", "Unknown Platform")
        url = market.get("url", market.get("link", ""))
        probability = market.get("probability", "N/A")
        volume = market.get("volume", "N/A")
        status = market.get("status", "N/A")
        end_date = market.get("end_date", market.get("resolution_date", "N/A"))
        formatted += (
            f"- {name}\n"
            f"  Platform: {platform}\n"
            f"  Probability: {probability}\n"
            f"  Volume: {volume}\n"
            f"  Status: {status}\n"
            f"  Ends: {end_date}\n"
            f"  URL: {url}\n"
        )
    return formatted

def get_web_search_results_from_openrouter(question: str) -> str:
    """
    Given a question string, use the OpenRouter completions API with a web-search-enabled model to get relevant news/info.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in environment.")
    url = "https://openrouter.ai/api/v1/completions"
    prompt = (
        "You are an assistant to a superforecaster. "
        "The superforecaster will give you a question they intend to forecast on. "
        "To be a great assistant, you generate a concise but detailed rundown of the most relevant news, "
        "including if the question would resolve Yes or No based on current information. "
        "You do not produce forecasts yourself.\n\n"
        f"Question: {question}"
    )
    payload = {
        "model": "openai/gpt-4o:online",
        "prompt": prompt,
        "max_tokens": 2048,
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(url, json=payload, headers=headers)
    print(f"DEBUG: Status code: {response.status_code}")
    print(f"DEBUG: Response text: {response.text}")
    if not response.ok:
        raise RuntimeError(f"OpenRouter completions API error: {response.text}")
    data = response.json()
    choices = data.get("choices")
    if not choices or not choices[0].get("text"):
        return "No web search results found."
    return choices[0]["text"].strip() 