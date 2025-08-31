import requests
from config import PERPLEXITY_API_KEY


def call_perplexity(question: str) -> str:
    """
    Call Perplexity AI API to get research results for the given question.
    """
    if not PERPLEXITY_API_KEY:
        return "Perplexity API key not provided."
    
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "content-type": "application/json",
    }
    payload = {
        "model": "llama-3.1-sonar-huge-128k-online",
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
                "content": question,
            },
        ],
    }
    
    try:
        response = requests.post(url=url, json=payload, headers=headers)
        if not response.ok:
            return f"Perplexity API error: {response.status_code} - {response.text}"
        
        content = response.json()["choices"][0]["message"]["content"]
        return content
        
    except Exception as e:
        return f"Error calling Perplexity API: {str(e)}"