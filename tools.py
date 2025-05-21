import os
import requests
from forecasting_tools import GeneralLlm, clean_indents, AskNewsSearcher
import logging
import re

logger = logging.getLogger("forecasting_tools.forecast_bots.forecast_bot")


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
    response = requests.request(
        "GET", base_url, params=params, headers=headers)
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
            # Store parsed volume for sorting
            market["_parsed_volume"] = vol_num
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
            f"- {name}\n\n"
            f"  Platform: {platform}\n\n"
            f"  Probability: {probability}\n\n"
            f"  Volume: {volume}\n\n"
            f"  Status: {status}\n\n"
            f"  Ends: {end_date}\n\n"
            f"  URL: {url}\n\n"
            "\n"  # Add a blank line between markets
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
        raise RuntimeError(
            f"OpenRouter completions API error: {response.text}")
    data = response.json()
    choices = data.get("choices")
    if not choices or not choices[0].get("text"):
        return "No web search results found."
    return choices[0]["text"].strip()


async def fermi_estimate_with_llm(question: str, llm: GeneralLlm) -> str:
    """
    Given a question string, use a configurable LLM to perform a Fermi estimation (back-of-the-envelope calculation).
    The LLM is prompted to break down the problem into logical steps, make explicit guesses, and document all reasoning.
    """
    prompt = clean_indents(
        f"""
        You are a professional forecaster skilled in Fermi estimation (back-of-the-envelope reasoning).
        Your task is to answer the following question using a Fermi estimation approach:

        {question}

        Instructions:
        - Break the problem down into smaller, logical components.
        - For each component, make explicit, reasonable guesses or estimates, and clearly state your assumptions.
        - Document every step and calculation in detail, showing your work.
        - Do not make any unstated assumptions; explain your reasoning for each guess.
        - Proceed step by step, combining your estimates to reach a final answer.
        - At the end, summarize your Fermi estimate and show the final calculation.
        """
    )
    response = await llm.invoke(prompt)
    return response


async def get_perplexity_research_from_openrouter(question: str, model_name: str = "openrouter/perplexity/sonar-pro", temperature: float = 0.1) -> str:
    """
    Calls the Perplexity model via OpenRouter for research purposes, using the same prompt as the template bot.
    You can specify the model (e.g., 'openrouter/perplexity/sonar-pro' or 'openrouter/perplexity/sonar-reasoning').
    """
    prompt = clean_indents(
        f"""
        You are an assistant to a superforecaster.
        The superforecaster will give you a question they intend to forecast on.
        To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
        You do not produce forecasts yourself.

        Question:
        {question}
        """
    )
    model = GeneralLlm(
        model=model_name,
        temperature=temperature,
    )
    response = await model.invoke(prompt)
    return response


def log_report_summary_returning_str(forecast_reports) -> str:
    # Import here to avoid circular import
    from forecasting_tools import clean_indents
    try:
        ForecastReport = __import__('forecasting_tools.forecast_bots.forecast_bot', fromlist=[
                                    'ForecastReport']).ForecastReport
    except Exception:
        ForecastReport = None
    valid_reports = [
        report
        for report in forecast_reports
        if ForecastReport and isinstance(report, ForecastReport)
    ]
    exceptions = [
        report
        for report in forecast_reports
        if isinstance(report, BaseException)
    ]
    minor_exceptions = [
        getattr(report, 'errors', None) for report in valid_reports if getattr(report, 'errors', None)
    ]

    full_summary = ""
    for report in valid_reports:
        question_summary = clean_indents(
            f"""
            URL: {report.question.page_url}
            Errors: {report.errors}
            <<<<<<<<<<<<<<<<<<<< Summary >>>>>>>>>>>>>>>>>>>>>
            {report.summary}

            <<<<<<<<<<<<<<<<<<<< First Rationales >>>>>>>>>>>>>>>>>>>>>
            {report.forecast_rationales.split('##')[1][:10000]}
            -------------------------------------------------------------------------------------------
        """
        )
        full_summary += question_summary + "\n"

    for report in forecast_reports:
        if ForecastReport and isinstance(report, ForecastReport):
            short_summary = f"✅ URL: {report.question.page_url} | Minor Errors: {len(report.errors)}"
        else:
            exception_message = (
                str(report)
                if len(str(report)) < 1000
                else f"{str(report)[:500]}...{str(report)[-500:]}"
            )
            short_summary = f"❌ Exception: {report.__class__.__name__} | Message: {exception_message}"
        full_summary += short_summary + "\n"
    logger.info(full_summary)

    if minor_exceptions:
        logger.error(
            f"{len(minor_exceptions)} minor exceptions occurred while forecasting: {minor_exceptions}"
        )
    if exceptions:
        raise RuntimeError(
            f"{len(exceptions)} errors occurred while forecasting: {exceptions}"
        )
    return full_summary


async def get_asknews_research(question: str) -> str:
    """
    Given a question string, use AskNewsSearcher to get formatted news results (async).
    Requires ASKNEWS_CLIENT_ID and ASKNEWS_SECRET in the environment.
    """
    if not (os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET")):
        raise ValueError(
            "ASKNEWS_CLIENT_ID and/or ASKNEWS_SECRET not set in environment.")
    searcher = AskNewsSearcher()
    return await searcher.get_formatted_news_async(question)


async def confirm_or_revise_prediction(message: str, question, llm: GeneralLlm) -> str:
    """
    Given a message (containing research/results and a prediction), a question object, and an LLM,
    prompt the LLM to confirm or revise its prediction, reiterating the required answer format based on the question type.
    Returns the LLM's response as a string.
    """
    # Determine question type and format instructions
    from forecasting_tools import clean_indents
    # Try to get the class name in a robust way
    qtype = type(question).__name__
    if qtype == "BinaryQuestion":
        format_instructions = (
            'If you agree with your prediction above, restate it. If you would like to revise it, provide a new rationale and answer.\n'
            'Format: Probability: ZZ% (0-100, no decimals, no space between number and % sign)'
        )
    elif qtype == "MultipleChoiceQuestion":
        options = getattr(question, 'options', None)
        options_str = f" in this order: {options}" if options else ""
        format_instructions = (
            f'If you agree with your probabilities above, restate them. If you would like to revise them, provide a new rationale and answer.\n'
            f'Format: Option_A: Probability_A\nOption_B: Probability_B\n...\nOption_N: Probability_N{options_str} (0-100, no decimals, no space between number and % sign)'
        )
    elif qtype == "NumericQuestion":
        unit = getattr(question, 'unit_of_measure',
                       'Not stated (please infer this)')
        format_instructions = (
            f'If you agree with your numeric distribution above, restate it. If you would like to revise it, provide a new rationale and answer.\n'
            f'Format:\nPercentile 10: XX\nPercentile 20: XX\nPercentile 40: XX\nPercentile 60: XX\nPercentile 80: XX\nPercentile 90: XX\nUnits: {unit}\n(Do not use scientific notation. Always start with the smallest value and increase. No decimals unless required by the question.)'
        )
    else:
        format_instructions = (
            'If you agree with your answer above, restate it. If you would like to revise it, provide a new rationale and answer.'
        )

    prompt = clean_indents(f"""
    {message}

    ---
    Do you agree with your prediction above, or would you like to revise it? Please answer below.
    {format_instructions}
    """)
    response = await llm.invoke(prompt)
    return response


def get_related_markets_raw(question: str) -> list[dict]:
    """
    Given a question string, use the Adjacent News API to find related markets and return them as a list of dictionaries.
    Only include markets with volume >= 1000, sorted by volume descending.
    """
    api_key = os.getenv("ADJACENT_NEWS_API_KEY")
    if not api_key:
        raise ValueError("ADJACENT_NEWS_API_KEY not set in environment.")
    base_url = "https://api.data.adj.news/api/search/query"
    params = {"q": question}
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.request(
        "GET", base_url, params=params, headers=headers)
    print(f"DEBUG: Status code: {response.status_code}")
    print(f"DEBUG: Response text: {response.text}")
    if not response.ok:
        raise RuntimeError(f"Adjacent News API error: {response.text}")
    data = response.json()
    print(f"DEBUG: Parsed data: {data}")
    if not data or "data" not in data or not data["data"]:
        return []
    filtered_markets = []
    for market in data["data"]:
        volume = market.get("volume", "N/A")
        try:
            vol_num = float(volume)
        except (ValueError, TypeError):
            vol_num = 0
        if vol_num >= 1000:
            # Store parsed volume for sorting
            market["_parsed_volume"] = vol_num
            filtered_markets.append(market)
    if not filtered_markets:
        return []
    # Sort by volume descending
    filtered_markets.sort(key=lambda m: m["_parsed_volume"], reverse=True)
    return filtered_markets


def format_markets(markets: list[dict]) -> str:
    """
    Format a list of market dictionaries into a readable string.
    """
    if not markets:
        return "No related markets found with volume >= 1000."
    formatted = "Related Markets from Adjacent News (volume >= 1000, sorted by volume):\n"
    for market in markets:
        name = market.get("name", market.get("question", "Unnamed Market"))
        platform = market.get("platform", "Unknown Platform")
        url = market.get("url", market.get("link", ""))
        probability = market.get("probability", "N/A")
        volume = market.get("volume", "N/A")
        status = market.get("status", "N/A")
        end_date = market.get("end_date", market.get("resolution_date", "N/A"))
        formatted += (
            f"- {name}\n\n"
            f"  Platform: {platform}\n\n"
            f"  Probability: {probability}\n\n"
            f"  Volume: {volume}\n\n"
            f"  Status: {status}\n\n"
            f"  Ends: {end_date}\n\n"
            f"  URL: {url}\n\n"
            "\n"  # Add a blank line between markets
        )
    return formatted


class IntegerExtractor:
    @staticmethod
    def extract_last_integer_value(
        text: str, max_value: int = 5, min_value: int = 1
    ) -> int:
        """
        Extract the last integer value from text that appears after "Score: ".
        The integer should be between min_value and max_value (inclusive).
        
        Args:
            text: The text to search for an integer
            max_value: Maximum allowed value (default: 5)
            min_value: Minimum allowed value (default: 1)
            
        Returns:
            The last integer found, clamped between min_value and max_value
            
        Raises:
            ValueError: If no integer is found or if text is empty
        """
        if not text or text.strip() == "":
            raise ValueError(
                "While trying to extract last integer value found that the text is None or an empty string"
            )
        assert (
            min_value <= max_value
        ), f"Max value {max_value} is not greater than or equal to min value {min_value}"
        
        # Look for integers after "Score: "
        matches = re.findall(r"Score:\s*(\d+)", text)
        if matches:
            # Return the last number found after "Score: "
            original_number = int(matches[-1])
            clamped_number = min(
                max_value, max(min_value, original_number)
            )
            assert (
                min_value <= clamped_number <= max_value
            ), f"Clamped number {clamped_number} is not between {min_value} and {max_value}"
            return int(clamped_number)
        else:
            raise ValueError(
                f"Could not extract integer from response. The text was: {text}"
            )


class FactsExtractor:
    @staticmethod
    def extract_facts(text: str) -> list[str]:
        """
        Extract a list of facts from text that appears after the "Key Facts" heading
        and before the next section.
        
        Args:
            text: The text to search for facts
            
        Returns:
            A list of facts, or empty list if none found
        """
        if not text or text.strip() == "":
            return []
            
        # Look for the facts section starting with "Key Facts"
        facts_section = re.search(r"Key Facts(.*?)(?=Follow Up Questions|$)", text, re.DOTALL)
        if not facts_section:
            return []
            
        # Split into individual facts
        facts_text = facts_section.group(1).strip()
        facts = []
        
        # Look for numbered or bulleted facts
        for line in facts_text.split('\n'):
            # Remove common bullet points and numbers
            line = re.sub(r'^[\d\.\-\*]+[\s]*', '', line.strip())
            if line and len(line) > 10:  # Only include non-empty lines with some content
                facts.append(line)
                
        return facts[:5]  # Return at most 5 facts


class FollowUpQuestionsExtractor:
    @staticmethod
    def extract_follow_up_questions(text: str) -> list[str]:
        """
        Extract a list of follow-up questions from text that appears after the "Follow Up Questions" heading
        and before the next section.
        
        Args:
            text: The text to search for follow-up questions
            
        Returns:
            A list of follow-up questions, or empty list if none found
        """
        if not text or text.strip() == "":
            return []
            
        # Look for the follow-up questions section starting with "Follow Up Questions"
        questions_section = re.search(r"Follow Up Questions(.*?)(?=Prediction Markets|$)", text, re.DOTALL)
        if not questions_section:
            return []
            
        # Split into individual questions
        questions_text = questions_section.group(1).strip()
        questions = []
        
        # Look for numbered or bulleted questions
        for line in questions_text.split('\n'):
            # Remove common bullet points and numbers
            line = re.sub(r'^[\d\.\-\*]+[\s]*', '', line.strip())
            if line and len(line) > 10 and line.endswith('?'):  # Only include non-empty lines that end with a question mark
                questions.append(line)
                
        return questions[:5]  # Return at most 5 questions
