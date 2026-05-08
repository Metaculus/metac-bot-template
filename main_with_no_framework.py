from __future__ import annotations

import asyncio
import datetime
import json
import math
import os
import re
from bisect import bisect_right
from dataclasses import dataclass
from typing import Sequence

import dotenv

dotenv.load_dotenv()

import forecasting_tools
import numpy as np
import requests
from asknews_sdk import AskNewsSDK
from openai import AsyncOpenAI

"""
This file provides a simple forecasting bot built from the ground up.
We provide this for people who want to dissect
it to build their own bot without using forecasting-tools.

This template assumes you are using a OpenAI model and have an OpenAI API key
You will also need a Metaculus API key, for posting questions to Metaculus
and a Perplexity or AskNews API key for online research

This is not a representative of the template bots used by Metaculus, as there are some
differences in implementation. The actual template bot (e.g. like main.py) has the following differences:
- An LLM now parses the final forecast output (rather than programmatic parsing)
- Support for nominal bounds was added (i.e. when there are discrete questions and normal upper/lower bounds are not as intuitive)
- Upper/Lower bounds are mentioned as suggestions (not ignored) when the bounds are open
- Group questions, conditional questions, and date questions are supported (these types are optional and won't be launched in Spring AIB)
- The research prompt mentions resolution criteria and fine print explicitly

We realize the below code could probably be cleaned up a bit in a few places
Though we are assuming most people will dissect it enough to make this not matter much

Note that this is code is given as-is and though we have have done basic testing
with this file it may be worth double checking key components locally.


"""


######################### CONSTANTS #########################
# Constants
SUBMIT_PREDICTION = True  # set to True to publish your predictions to Metaculus
USE_EXAMPLE_QUESTIONS = False  # set to True to forecast example questions rather than the tournament questions
NUM_RUNS_PER_QUESTION = (
    3  # The median forecast is taken between NUM_RUNS_PER_QUESTION runs
)
SKIP_PREVIOUSLY_FORECASTED_QUESTIONS = False

# Environment variables
# You only need *either* Exa or Perplexity or AskNews keys for online research
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_SECRET = os.getenv("ASKNEWS_SECRET")
EXA_API_KEY = os.getenv("EXA_API_KEY")
OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)  # You'll also need the OpenAI API Key if you want to use the Exa Smart Searcher

# The tournament IDs below can be used for testing your bot.
Q4_2024_AI_BENCHMARKING_ID = 32506
Q1_2025_AI_BENCHMARKING_ID = 32627
FALL_2025_AI_BENCHMARKING_ID = "fall-aib-2025"
SPRING_2026_AI_BENCHMARKING_ID = "spring-aib-2026"
SUMMER_2026_AI_BENCHMARKING_ID = 33022  # https://www.metaculus.com/tournament/summer-futureeval-2026/

CURRENT_MINIBENCH_ID = "minibench"

Q4_2024_QUARTERLY_CUP_ID = 3672
Q1_2025_QUARTERLY_CUP_ID = 32630
CURRENT_METACULUS_CUP_ID = None # TBD (Use the slug from the Metaculus Cup URL)

AXC_2025_TOURNAMENT_ID = 32564
AI_2027_TOURNAMENT_ID = "ai-2027"

TOURNAMENT_ID = SUMMER_2026_AI_BENCHMARKING_ID

# The example questions can be used for testing your bot. (note that question and post id are not always the same)
EXAMPLE_QUESTIONS = [  # (question_id, post_id)
    (
        578,
        578,
    ),  # Human Extinction - Binary - https://www.metaculus.com/questions/578/human-extinction-by-2100/
    (
        14333,
        14333,
    ),  # Age of Oldest Human - Numeric - https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/
    (
        22427,
        22427,
    ),  # Number of New Leading AI Labs - Multiple Choice - https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/
    (
        38195,
        38880,
    ),  # Number of US Labor Strikes Due to AI in 2029 - Discrete - https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/
]


######################### HELPER FUNCTIONS #########################

# @title Helper functions
AUTH_HEADERS = {"headers": {"Authorization": f"Token {METACULUS_TOKEN}"}}
API_BASE_URL = "https://www.metaculus.com/api"


def post_question_comment(post_id: int, comment_text: str) -> None:
    """
    Post a comment on the question page as the bot user.
    """

    response = requests.post(
        f"{API_BASE_URL}/comments/create/",
        json={
            "text": comment_text,
            "parent": None,
            "included_forecast": True,
            "is_private": True,
            "on_post": post_id,
        },
        **AUTH_HEADERS,  # type: ignore
    )
    if not response.ok:
        raise RuntimeError(response.text)


def post_question_prediction(question_id: int, forecast_payload: dict) -> None:
    """
    Post a forecast on a question.
    """
    url = f"{API_BASE_URL}/questions/forecast/"
    response = requests.post(
        url,
        json=[
            {
                "question": question_id,
                **forecast_payload,
            },
        ],
        **AUTH_HEADERS,  # type: ignore
    )
    print(f"Prediction Post status code: {response.status_code}")
    if not response.ok:
        raise RuntimeError(response.text)


def create_forecast_payload(
    forecast: float | dict[str, float] | list[float],
    question_type: str,
) -> dict:
    """
    Accepts a forecast and generates the api payload in the correct format.

    If the question is binary, forecast must be a float.
    If the question is multiple choice, forecast must be a dictionary that
      maps question.options labels to floats.
    If the question is numeric, forecast must be a dictionary that maps
      quartiles or percentiles to datetimes, or a 201 value cdf.
    """
    if question_type == "binary":
        return {
            "probability_yes": forecast,
            "probability_yes_per_category": None,
            "continuous_cdf": None,
        }
    if question_type == "multiple_choice":
        return {
            "probability_yes": None,
            "probability_yes_per_category": forecast,
            "continuous_cdf": None,
        }
    # numeric or date
    return {
        "probability_yes": None,
        "probability_yes_per_category": None,
        "continuous_cdf": forecast,
    }


def list_posts_from_tournament(
    tournament_id: int | str = TOURNAMENT_ID, offset: int = 0, count: int = 50
) -> list[dict]:
    """
    List (all details) {count} posts from the {tournament_id}
    """
    url_qparams = {
        "limit": count,
        "offset": offset,
        "order_by": "-hotness",
        "forecast_type": ",".join(
            [
                "binary",
                "multiple_choice",
                "numeric",
                "discrete",
            ]
        ),
        "tournaments": [tournament_id],
        "statuses": "open",
        "include_description": "true",
    }
    url = f"{API_BASE_URL}/posts/"
    response = requests.get(url, **AUTH_HEADERS, params=url_qparams)  # type: ignore
    if not response.ok:
        raise Exception(response.text)
    data = json.loads(response.content)
    return data


def get_open_question_ids_from_tournament() -> list[tuple[int, int]]:
    posts = list_posts_from_tournament()

    post_dict = dict()
    for post in posts["results"]:
        if question := post.get("question"):
            # single question post
            post_dict[post["id"]] = [question]

    open_question_id_post_id = []  # [(question_id, post_id)]
    for post_id, questions in post_dict.items():
        for question in questions:
            if question.get("status") == "open":
                print(
                    f"ID: {question['id']}\nQ: {question['title']}\nCloses: "
                    f"{question['scheduled_close_time']}"
                )
                open_question_id_post_id.append((question["id"], post_id))

    return open_question_id_post_id


def get_post_details(post_id: int) -> dict:
    """
    Get all details about a post from the Metaculus API.
    """
    url = f"{API_BASE_URL}/posts/{post_id}/"
    print(f"Getting details for {url}")
    response = requests.get(
        url,
        **AUTH_HEADERS,  # type: ignore
    )
    if not response.ok:
        raise Exception(response.text)
    details = json.loads(response.content)
    return details


CONCURRENT_REQUESTS_LIMIT = 5
llm_rate_limiter = asyncio.Semaphore(CONCURRENT_REQUESTS_LIMIT)


async def call_llm(prompt: str, model: str = "gpt-4o", temperature: float = 0.3) -> str:
    """
    Makes a streaming completion request to OpenAI's API with concurrent request limiting.
    """

    # Remove the base_url parameter to call the OpenAI API directly
    # Also checkout the package 'litellm' for one function that can call any model from any provider
    # Also checkout OpenRouter for allowing one API key for many providers (especially powerful if combined with litellm)
    client = AsyncOpenAI()

    async with llm_rate_limiter:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            stream=False,
        )
        answer = response.choices[0].message.content
        if answer is None:
            raise ValueError("No answer returned from LLM")
        return answer


def run_research(question: str) -> str:
    research = ""
    if ASKNEWS_CLIENT_ID and ASKNEWS_SECRET:
        research = call_asknews(question)
    elif EXA_API_KEY:
        research = call_exa_smart_searcher(question)
    elif PERPLEXITY_API_KEY:
        research = call_perplexity(question)
    else:
        research = "No research done"

    print(
        f"########################\nResearch Found:\n{research}\n########################"
    )

    return research


def call_perplexity(question: str) -> str:
    url = "https://api.perplexity.ai/chat/completions"
    api_key = PERPLEXITY_API_KEY
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {api_key}",
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
    response = requests.post(url=url, json=payload, headers=headers)
    if not response.ok:
        raise Exception(response.text)
    content = response.json()["choices"][0]["message"]["content"]
    return content


def call_exa_smart_searcher(question: str) -> str:
    if OPENAI_API_KEY is None:
        searcher = forecasting_tools.ExaSearcher(
            include_highlights=True,
            num_results=10,
        )
        highlights = asyncio.run(
            searcher.invoke_for_highlights_in_relevance_order(question)
        )
        prioritized_highlights = highlights[:10]
        combined_highlights = ""
        for i, highlight in enumerate(prioritized_highlights):
            combined_highlights += f'[Highlight {i+1}]:\nTitle: {highlight.source.title}\nURL: {highlight.source.url}\nText: "{highlight.highlight_text}"\n\n'
        response = combined_highlights
    else:
        searcher = forecasting_tools.SmartSearcher(
            temperature=0,
            num_searches_to_run=2,
            num_sites_per_search=10,
        )
        prompt = (
            "You are an assistant to a superforecaster. The superforecaster will give"
            "you a question they intend to forecast on. To be a great assistant, you generate"
            "a concise but detailed rundown of the most relevant news, including if the question"
            "would resolve Yes or No based on current information. You do not produce forecasts yourself."
            f"\n\nThe question is: {question}"
        )
        response = asyncio.run(searcher.invoke(prompt))
        assert response is not None

    return response


def call_asknews(question: str) -> str:
    """
    Use the AskNews `news` endpoint to get news context for your query.
    The full API reference can be found here: https://docs.asknews.app/en/reference#get-/v1/news/search
    """
    ask = AskNewsSDK(
        client_id=ASKNEWS_CLIENT_ID, client_secret=ASKNEWS_SECRET, scopes=set(["news"])
    )

    # get the latest news related to the query (within the past 48 hours)
    hot_response = ask.news.search_news(
        query=question,  # your natural language query
        n_articles=6,  # control the number of articles to include in the context, originally 5
        return_type="both",
        strategy="latest news",  # enforces looking at the latest news only
    )

    # get context from the "historical" database that contains a news archive going back to 2023
    historical_response = ask.news.search_news(
        query=question,
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


############### BINARY ###############
# @title Binary prompt & functions

# This section includes functionality for binary questions.

BINARY_PROMPT_TEMPLATE = """
You are a professional forecaster interviewing for a job.

Your interview question is:
{title}

Question background:
{background}


This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
{resolution_criteria}

{fine_print}


Your research assistant says:
{summary_report}

Today is {today}.

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The status quo outcome if nothing changed.
(c) A brief description of a scenario that results in a No outcome.
(d) A brief description of a scenario that results in a Yes outcome.

You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

The last thing you write is your final answer as: "Probability: ZZ%", 0-100
"""


def extract_probability_from_response_as_percentage_not_decimal(
    forecast_text: str,
) -> float:
    matches = re.findall(r"(\d+)%", forecast_text)
    if matches:
        # Return the last number found before a '%'
        number = int(matches[-1])
        number = min(99, max(1, number))  # clamp the number between 1 and 99
        return number
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")


async def get_binary_gpt_prediction(
    question_details: dict, num_runs: int
) -> tuple[float, str]:

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]
    question_type = question_details["type"]

    summary_report = run_research(title)

    content = BINARY_PROMPT_TEMPLATE.format(
        title=title,
        today=today,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        summary_report=summary_report,
    )

    async def get_rationale_and_probability(content: str) -> tuple[float, str]:
        rationale = await call_llm(content)

        probability = extract_probability_from_response_as_percentage_not_decimal(
            rationale
        )
        comment = (
            f"Extracted Probability: {probability}%\n\nGPT's Answer: "
            f"{rationale}\n\n\n"
        )
        return probability, comment

    probability_and_comment_pairs = await asyncio.gather(
        *[get_rationale_and_probability(content) for _ in range(num_runs)]
    )
    comments = [pair[1] for pair in probability_and_comment_pairs]
    final_comment_sections = [
        f"## Rationale {i+1}\n{comment}" for i, comment in enumerate(comments)
    ]
    probabilities = [pair[0] for pair in probability_and_comment_pairs]
    median_probability = float(np.median(probabilities)) / 100

    final_comment = f"Median Probability: {median_probability}\n\n" + "\n\n".join(
        final_comment_sections
    )
    return median_probability, final_comment


####################### NUMERIC ###############
# @title Numeric prompt & functions
#
# This section includes functionality for numeric questions.
# The prompt emphasizes the importance of putting percentile values in the correct order, and the code includes functions for generating and standardizing CDFs based on LLM output.

NUMERIC_PROMPT_TEMPLATE = """
You are a professional forecaster interviewing for a job.

Your interview question is:
{title}

Background:
{background}

{resolution_criteria}

{fine_print}

Units for answer: {units}

Your research assistant says:
{summary_report}

Today is {today}.

The scale ranges from {lower_bound} to {upper_bound}{log_note}.
{lower_bound_message}
{upper_bound_message}

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The outcome if nothing changed.
(c) The outcome if the current trend continued.
(d) The expectations of experts and markets.
(e) A brief description of an unexpected scenario that results in a low outcome.
(f) A brief description of an unexpected scenario that results in a high outcome.

You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns. Never use scientific notation.

The last thing you write is your forecast as a JSON code block. Use keys of the form "p<N>" where N is between 1 and 99 (e.g. "p5", "p25", "p50", "p75", "p95"). Provide at least 5 percentiles spanning a wide interval. Values must be strictly increasing. Example:
```json
{{
  "p5": {ex_p5},
  "p25": {ex_p25},
  "p50": {ex_p50},
  "p75": {ex_p75},
  "p95": {ex_p95}
}}
```
"""


@dataclass
class Scaling:

    range_min: float
    range_max: float
    zero_point: float | None
    open_lower_bound: bool
    open_upper_bound: bool
    inbound_outcome_count: int | None = None


def unscale_value(scaled: float, scaling: Scaling) -> float:
    lo, hi, zp = scaling.range_min, scaling.range_max, scaling.zero_point
    if zp is None:
        return (scaled - lo) / (hi - lo)
    deriv_ratio = (hi - zp) / (lo - zp)
    return (
        np.log((scaled - lo) * (deriv_ratio - 1) + (hi - lo)) - np.log(hi - lo)
    ) / np.log(deriv_ratio)


class MonotoneCubicInterpolator:
    """Monotone piecewise cubic Hermite interpolation (PCHIP / Fritsch-Carlson).

    Tangents are constrained so the interpolant cannot overshoot or oscillate
    between knots — guarantees monotone output for monotone input. Ideal for
    CDF interpolation.
    """

    def __init__(self, x: Sequence[float], y: Sequence[float]):
        x = list(map(float, x))
        y = list(map(float, y))
        n = len(x)
        if n < 2:
            raise ValueError("Need at least two points.")
        for i in range(n - 1):
            if not (x[i + 1] > x[i]):
                raise ValueError("x must be strictly increasing.")

        h = [x[i + 1] - x[i] for i in range(n - 1)]
        d = [(y[i + 1] - y[i]) / h[i] for i in range(n - 1)]

        m = [0.0] * n
        m[0] = d[0]
        m[-1] = d[-1]
        for i in range(1, n - 1):
            if d[i - 1] * d[i] <= 0:
                m[i] = 0.0
            else:
                w1 = 2 * h[i] + h[i - 1]
                w2 = h[i] + 2 * h[i - 1]
                m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])

        # Fritsch-Carlson circle condition: prevent overshoot at endpoints.
        for i in range(n - 1):
            if abs(d[i]) < 1e-14:
                m[i] = m[i + 1] = 0.0
                continue
            alpha = m[i] / d[i]
            beta = m[i + 1] / d[i]
            if alpha < 0:
                m[i] = 0.0
                alpha = 0.0
            if beta < 0:
                m[i + 1] = 0.0
                beta = 0.0
            r2 = alpha**2 + beta**2
            if r2 > 9.0:
                tau = 3.0 / math.sqrt(r2)
                m[i] = tau * alpha * d[i]
                m[i + 1] = tau * beta * d[i]

        self.x = x
        self.y = y
        self.m = m
        self.h = h

    def __call__(self, xp: float) -> float:
        x, y, m, h = self.x, self.y, self.m, self.h
        if xp <= x[0]:
            return y[0]
        if xp >= x[-1]:
            return y[-1]
        i = min(bisect_right(x, xp) - 1, len(x) - 2)
        t = (xp - x[i]) / h[i]
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        return h00 * y[i] + h10 * h[i] * m[i] + h01 * y[i + 1] + h11 * h[i] * m[i + 1]


def _cdf_at_boundary(
    percentile_items: list[tuple[float, float]],
    boundary: float,
) -> float:
    """Linear interpolation of the CDF at a boundary value."""
    if boundary < percentile_items[0][1]:
        return percentile_items[0][0] / 2.0
    if boundary > percentile_items[-1][1]:
        return (percentile_items[-1][0] + 1.0) / 2.0
    for i, (frac, val) in enumerate(percentile_items):
        if val == boundary:
            return frac
        if i + 1 < len(percentile_items):
            frac_next, val_next = percentile_items[i + 1]
            if val < boundary < val_next:
                t = (boundary - val) / (val_next - val)
                return frac + t * (frac_next - frac)
    return percentile_items[-1][0]


def _infer_below_above(
    percentile_items: list[tuple[float, float]],
    scaling: Scaling,
) -> tuple[float, float]:
    """Probability mass outside open bounds (0.0 for closed bounds)."""
    if not scaling.open_lower_bound:
        below: float = 0.0
    elif percentile_items:
        below = _cdf_at_boundary(percentile_items, float(scaling.range_min))
    else:
        below = 0.0

    if not scaling.open_upper_bound:
        above: float = 0.0
    elif percentile_items:
        above = 1.0 - _cdf_at_boundary(percentile_items, float(scaling.range_max))
    else:
        above = 0.0

    return below, above


def cdf_to_pmf(cdf: list[float]) -> list[float]:
    pmf = [cdf[0]]
    for i in range(1, len(cdf)):
        pmf.append(max(0.0, cdf[i] - cdf[i - 1]))
    pmf.append(max(0.0, 1.0 - cdf[-1]))
    return pmf


def pmf_to_cdf(pmf: list[float]) -> list[float]:
    cdf = []
    cumsum = 0.0
    for v in pmf[:-1]:
        cumsum += v
        cdf.append(cumsum)
    return cdf


def generate_continuous_cdf(
    percentiles: dict[float, float],
    scaling: Scaling,
    below_lower_bound: float | None = None,
    above_upper_bound: float | None = None,
) -> list[float]:
    """Generate a CDF for a continuous question from a set of percentiles.

    Args:
        percentiles: mapping of percentile (0–1) to value on the question's scale.
        scaling: question scaling info.
        below_lower_bound: probability mass below the lower bound.
        above_upper_bound: probability mass above the upper bound.

    Returns:
        A list of `inbound_outcome_count + 1` CDF values.
    """
    points: list[tuple[float, float]] = []
    if below_lower_bound is not None:
        points.append((0.0, below_lower_bound))
    if above_upper_bound is not None:
        points.append((1.0, 1.0 - above_upper_bound))
    for pct, scaled in percentiles.items():
        points.append((unscale_value(scaled, scaling), pct))
    points.sort()

    # Average y values for any tied x values so the spline gets strictly
    # increasing x coordinates.
    deduped: list[tuple[float, float]] = []
    i = 0
    while i < len(points):
        j = i + 1
        while j < len(points) and points[j][0] == points[i][0]:
            j += 1
        x = points[i][0]
        y = sum(p[1] for p in points[i:j]) / (j - i)
        deduped.append((x, y))
        i = j
    points = deduped

    first, last = points[0], points[-1]
    if first[0] > 0.0 or last[0] < 1.0:
        raise ValueError("Percentiles must encompass the full range of the question")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mono = MonotoneCubicInterpolator(xs, ys)

    n = scaling.inbound_outcome_count or 200
    cdf = [round(mono(i / n), 10) for i in range(n + 1)]
    if any(not (0.0 <= F <= 1.0) for F in cdf):
        raise ValueError(f"Interpolated CDF values must be in [0, 1]. Got: {cdf}")
    if any(np.isnan(f) for f in cdf):
        raise ValueError(f"Interpolated CDF contains NaN values: {cdf}")
    return cdf


def standardize_cdf(cdf: list[float], scaling: Scaling) -> list[float]:
    """Standardize a CDF so it satisfies Metaculus API constraints:

    - no mass outside closed bounds (rescaled accordingly)
    - at least 0.1% mass outside open bounds
    - minimum increase per step
    - maximum step capped to avoid PMF spikes
    """
    n = scaling.inbound_outcome_count or 200
    default_n = 200

    arr = np.asarray(cdf, dtype=float)
    if not arr.size:
        return []

    # PCHIP can mildly overshoot when an in-range high percentile is paired
    # with a small above-bound anchor. Clip to monotone before standardizing
    # so the minimum-step guarantee isn't overwhelmed by a downward jump.
    arr = np.maximum.accumulate(arr)

    scale_lower_to = 0.0 if scaling.open_lower_bound else arr[0]
    scale_upper_to = 1.0 if scaling.open_upper_bound else arr[-1]
    inbound_mass = scale_upper_to - scale_lower_to

    def standardize(F: float, location: float) -> float:
        rescaled = (F - scale_lower_to) / inbound_mass
        if scaling.open_lower_bound and scaling.open_upper_bound:
            return 0.988 * rescaled + 0.01 * location + 0.001
        elif scaling.open_lower_bound:
            return 0.989 * rescaled + 0.01 * location + 0.001
        elif scaling.open_upper_bound:
            return 0.989 * rescaled + 0.01 * location
        return 0.99 * rescaled + 0.01 * location

    for i, value in enumerate(arr):
        arr[i] = standardize(value, i / (len(arr) - 1))

    pmf = np.array(cdf_to_pmf(arr.tolist()))
    cap = 0.2 * (default_n / n)

    def cap_pmf(scale: float) -> np.ndarray:
        return np.concatenate([pmf[:1], np.minimum(cap, scale * pmf[1:-1]), pmf[-1:]])

    def capped_sum(scale: float) -> float:
        return float(cap_pmf(scale).sum())

    lo_s = hi_s = scale = 1.0
    while capped_sum(hi_s) < 1.0:
        hi_s *= 1.2
    for _ in range(100):
        scale = 0.5 * (lo_s + hi_s)
        s = capped_sum(scale)
        if s < 1.0:
            lo_s = scale
        else:
            hi_s = scale
        if s == 1.0 or (hi_s - lo_s) < 2e-5:
            break

    pmf = cap_pmf(scale)
    pmf[1:-1] *= (arr[-1] - arr[0]) / pmf[1:-1].sum()
    return np.round(pmf_to_cdf(pmf.tolist()), 10).tolist()


def extract_percentiles_from_response(forecast_text: str) -> dict[float, float]:
    """Parse a JSON {p<N>: value} block from the LLM output.

    Returns a dict mapping percentile fraction (0–1) → value. Values are
    sort-fallback-corrected: if the LLM emits non-monotone values, both
    percentiles and values are sorted independently and re-paired so the
    result is monotone in both dimensions. This trades fidelity to the LLM's
    intent for robustness against off-by-one mistakes; a strict raise would
    instead reject the whole run.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", forecast_text, re.DOTALL)
    if fenced:
        raw = fenced.group(1)
    else:
        bare = re.search(r"\{.*\}", forecast_text, re.DOTALL)
        if not bare:
            raise ValueError(
                f"No JSON object found in response: {forecast_text[:200]!r}"
            )
        raw = bare.group(0)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON parse error: {exc}. Raw: {raw[:200]!r}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")

    items: list[tuple[float, float]] = []
    for key, raw_value in data.items():
        m = re.fullmatch(r"p([1-9][0-9]?)", str(key).strip())
        if not m:
            continue
        n = int(m.group(1))
        if not (1 <= n <= 99):
            continue
        try:
            items.append((n / 100.0, float(raw_value)))
        except (TypeError, ValueError):
            continue

    if not items:
        raise ValueError(
            f"No valid percentile keys (p1–p99) in response: {forecast_text[:200]!r}"
        )

    items.sort()
    sorted_values = sorted(v for _, v in items)
    items = [(p, v) for (p, _), v in zip(items, sorted_values)]

    return dict(items)


async def get_numeric_gpt_prediction(
    question_details: dict, num_runs: int
) -> tuple[list[float], str]:

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]
    question_type = question_details["type"]
    raw_scaling = question_details["scaling"]
    open_upper_bound = question_details["open_upper_bound"]
    open_lower_bound = question_details["open_lower_bound"]
    unit_of_measure = (
        question_details["unit"]
        if question_details["unit"]
        else "Not stated (please infer this)"
    )
    upper_bound = raw_scaling["range_max"]
    lower_bound = raw_scaling["range_min"]
    zero_point = raw_scaling["zero_point"]
    if question_type == "discrete":
        outcome_count = raw_scaling["inbound_outcome_count"]
    else:
        outcome_count = 200

    scaling = Scaling(
        range_min=lower_bound,
        range_max=upper_bound,
        zero_point=zero_point,
        open_lower_bound=open_lower_bound,
        open_upper_bound=open_upper_bound,
        inbound_outcome_count=outcome_count,
    )

    if open_upper_bound:
        upper_bound_message = (
            f"The question creator thinks the value is likely not higher than {upper_bound}."
        )
    else:
        upper_bound_message = f"The outcome cannot be higher than {upper_bound}."
    if open_lower_bound:
        lower_bound_message = (
            f"The question creator thinks the value is likely not lower than {lower_bound}."
        )
    else:
        lower_bound_message = f"The outcome cannot be lower than {lower_bound}."

    log_note = " (logarithmic scale)" if zero_point is not None else ""
    span = upper_bound - lower_bound

    def example_at(fraction: float) -> str:
        return f"{lower_bound + fraction * span:g}"

    summary_report = run_research(title)

    content = NUMERIC_PROMPT_TEMPLATE.format(
        title=title,
        today=today,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        summary_report=summary_report,
        lower_bound_message=lower_bound_message,
        upper_bound_message=upper_bound_message,
        units=unit_of_measure,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        log_note=log_note,
        ex_p5=example_at(0.05),
        ex_p25=example_at(0.25),
        ex_p50=example_at(0.5),
        ex_p75=example_at(0.75),
        ex_p95=example_at(0.95),
    )

    async def ask_llm_to_get_cdf(content: str) -> tuple[list[float], str]:
        rationale = await call_llm(content)
        percentiles = extract_percentiles_from_response(rationale)

        sorted_items = sorted(percentiles.items())
        below, above = _infer_below_above(sorted_items, scaling)
        in_bounds = {
            p: v for p, v in sorted_items if lower_bound <= v <= upper_bound
        }

        cdf = generate_continuous_cdf(
            in_bounds,
            scaling,
            below_lower_bound=below,
            above_upper_bound=above,
        )
        cdf = standardize_cdf(cdf, scaling)

        comment = (
            f"Extracted percentiles: {percentiles}\n"
            f"Inferred below/above mass: {below:.4f} / {above:.4f}\n\n"
            f"GPT's Answer: {rationale}\n\n\n"
        )
        return cdf, comment

    cdf_and_comment_pairs = await asyncio.gather(
        *[ask_llm_to_get_cdf(content) for _ in range(num_runs)]
    )
    comments = [pair[1] for pair in cdf_and_comment_pairs]
    final_comment_sections = [
        f"## Rationale {i+1}\n{comment}" for i, comment in enumerate(comments)
    ]
    cdfs: list[list[float]] = [pair[0] for pair in cdf_and_comment_pairs]
    all_cdfs = np.array(cdfs)
    median_cdf: list[float] = np.median(all_cdfs, axis=0).tolist()

    final_comment = f"Median CDF: `{str(median_cdf)[:100]}...`\n\n" + "\n\n".join(
        final_comment_sections
    )
    return median_cdf, final_comment


########################## MULTIPLE CHOICE ###############
# @title Multiple Choice prompt & functions

MULTIPLE_CHOICE_PROMPT_TEMPLATE = """
You are a professional forecaster interviewing for a job.

Your interview question is:
{title}

The options are: {options}


Background:
{background}

{resolution_criteria}

{fine_print}


Your research assistant says:
{summary_report}

Today is {today}.

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The status quo outcome if nothing changed.
(c) A description of an scenario that results in an unexpected outcome.

You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

The last thing you write is your final probabilities for the N options in this order {options} as:
Option_A: Probability_A
Option_B: Probability_B
...
Option_N: Probability_N
"""


def extract_option_probabilities_from_response(forecast_text: str, options) -> float:

    # Helper function that returns a list of tuples with numbers for all lines with Percentile
    def extract_option_probabilities(text):

        # Number extraction pattern
        number_pattern = r"-?\d+(?:,\d{3})*(?:\.\d+)?"

        results = []

        # Iterate through each line in the text
        for line in text.split("\n"):
            # Extract all numbers from the line
            numbers = re.findall(number_pattern, line)
            numbers_no_commas = [num.replace(",", "") for num in numbers]
            # Convert strings to float or int
            numbers = [
                float(num) if "." in num else int(num) for num in numbers_no_commas
            ]
            # Add the tuple of numbers to results
            if len(numbers) >= 1:
                last_number = numbers[-1]
                results.append(last_number)

        return results

    option_probabilities = extract_option_probabilities(forecast_text)

    NUM_OPTIONS = len(options)

    if len(option_probabilities) > 0:
        # return the last NUM_OPTIONS items
        return option_probabilities[-NUM_OPTIONS:]
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")


def generate_multiple_choice_forecast(options, option_probabilities) -> dict:
    """
    Returns: dict corresponding to the probabilities of each option.
    """

    # confirm that there is a probability for each option
    if len(options) != len(option_probabilities):
        raise ValueError(
            f"Number of options ({len(options)}) does not match number of probabilities ({len(option_probabilities)})"
        )

    # Ensure we are using decimals
    total_sum = sum(option_probabilities)
    decimal_list = [x / total_sum for x in option_probabilities]

    def normalize_list(float_list):
        # Step 1: Clamp values
        clamped_list = [max(min(x, 0.99), 0.01) for x in float_list]

        # Step 2: Calculate the sum of all elements
        total_sum = sum(clamped_list)

        # Step 3: Normalize the list so that all elements add up to 1
        normalized_list = [x / total_sum for x in clamped_list]

        # Step 4: Adjust for any small floating-point errors
        adjustment = 1.0 - sum(normalized_list)
        normalized_list[-1] += adjustment

        return normalized_list

    normalized_option_probabilities = normalize_list(decimal_list)

    probability_yes_per_category = {}
    for i in range(len(options)):
        probability_yes_per_category[options[i]] = normalized_option_probabilities[i]

    return probability_yes_per_category


async def get_multiple_choice_gpt_prediction(
    question_details: dict,
    num_runs: int,
) -> tuple[dict[str, float], str]:

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]
    question_type = question_details["type"]
    options = question_details["options"]

    summary_report = run_research(title)

    content = MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(
        title=title,
        today=today,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        summary_report=summary_report,
        options=options,
    )

    async def ask_llm_for_multiple_choice_probabilities(
        content: str,
    ) -> tuple[dict[str, float], str]:
        rationale = await call_llm(content)

        option_probabilities = extract_option_probabilities_from_response(
            rationale, options
        )

        comment = (
            f"EXTRACTED_PROBABILITIES: {option_probabilities}\n\nGPT's Answer: "
            f"{rationale}\n\n\n"
        )

        probability_yes_per_category = generate_multiple_choice_forecast(
            options, option_probabilities
        )
        return probability_yes_per_category, comment

    probability_yes_per_category_and_comment_pairs = await asyncio.gather(
        *[ask_llm_for_multiple_choice_probabilities(content) for _ in range(num_runs)]
    )
    comments = [pair[1] for pair in probability_yes_per_category_and_comment_pairs]
    final_comment_sections = [
        f"## Rationale {i+1}\n{comment}" for i, comment in enumerate(comments)
    ]
    probability_yes_per_category_dicts: list[dict[str, float]] = [
        pair[0] for pair in probability_yes_per_category_and_comment_pairs
    ]
    average_probability_yes_per_category: dict[str, float] = {}
    for option in options:
        probabilities_for_current_option: list[float] = [
            dict[option] for dict in probability_yes_per_category_dicts
        ]
        average_probability_yes_per_category[option] = sum(
            probabilities_for_current_option
        ) / len(probabilities_for_current_option)

    final_comment = (
        f"Average Probability Yes Per Category: `{average_probability_yes_per_category}`\n\n"
        + "\n\n".join(final_comment_sections)
    )
    return average_probability_yes_per_category, final_comment


################### FORECASTING ###################
def forecast_is_already_made(post_details: dict) -> bool:
    """
    Check if a forecast has already been made by looking at my_forecasts in the question data.

    question.my_forecasts.latest.forecast_values has the following values for each question type:
    Binary: [probability for no, probability for yes]
    Numeric: [cdf value 1, cdf value 2, ..., cdf value 201]
    Multiple Choice: [probability for option 1, probability for option 2, ...]
    """
    try:
        forecast_values = post_details["question"]["my_forecasts"]["latest"][
            "forecast_values"
        ]
        return forecast_values is not None
    except Exception:
        return False


async def forecast_individual_question(
    question_id: int,
    post_id: int,
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
) -> str:
    post_details = get_post_details(post_id)
    question_details = post_details["question"]
    title = question_details["title"]
    question_type = question_details["type"]

    summary_of_forecast = ""
    summary_of_forecast += (
        f"-----------------------------------------------\nQuestion: {title}\n"
    )
    summary_of_forecast += f"URL: https://www.metaculus.com/questions/{post_id}/\n"

    if question_type == "multiple_choice":
        options = question_details["options"]
        summary_of_forecast += f"options: {options}\n"

    if (
        forecast_is_already_made(post_details)
        and skip_previously_forecasted_questions == True
    ):
        summary_of_forecast += f"Skipped: Forecast already made\n"
        return summary_of_forecast

    if question_type == "binary":
        forecast, comment = await get_binary_gpt_prediction(
            question_details, num_runs_per_question
        )
    elif question_type == "numeric":
        forecast, comment = await get_numeric_gpt_prediction(
            question_details, num_runs_per_question
        )
    elif question_type == "discrete":
        forecast, comment = await get_numeric_gpt_prediction(
            question_details, num_runs_per_question
        )
    elif question_type == "multiple_choice":
        forecast, comment = await get_multiple_choice_gpt_prediction(
            question_details, num_runs_per_question
        )
    else:
        raise ValueError(f"Unknown question type: {question_type}")

    print(
        f"-----------------------------------------------\nPost {post_id} Question {question_id}:\n"
    )
    print(f"Forecast for post {post_id} (question {question_id}):\n{forecast}")
    print(f"Comment for post {post_id} (question {question_id}):\n{comment}")

    if question_type == "numeric" or question_type == "discrete":
        summary_of_forecast += f"Forecast: {str(forecast)[:200]}...\n"
    else:
        summary_of_forecast += f"Forecast: {forecast}\n"

    summary_of_forecast += f"Comment:\n```\n{comment[:200]}...\n```\n\n"

    if submit_prediction == True:
        forecast_payload = create_forecast_payload(forecast, question_type)
        post_question_prediction(question_id, forecast_payload)
        post_question_comment(post_id, comment)
        summary_of_forecast += "Posted: Forecast was posted to Metaculus.\n"

    return summary_of_forecast


async def forecast_questions(
    open_question_id_post_id: list[tuple[int, int]],
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
) -> None:
    forecast_tasks = [
        forecast_individual_question(
            question_id,
            post_id,
            submit_prediction,
            num_runs_per_question,
            skip_previously_forecasted_questions,
        )
        for question_id, post_id in open_question_id_post_id
    ]
    forecast_summaries = await asyncio.gather(*forecast_tasks, return_exceptions=True)
    print("\n", "#" * 100, "\nForecast Summaries\n", "#" * 100)

    errors = []
    for question_id_post_id, forecast_summary in zip(
        open_question_id_post_id, forecast_summaries
    ):
        question_id, post_id = question_id_post_id
        if isinstance(forecast_summary, Exception):
            print(
                f"-----------------------------------------------\nPost {post_id} Question {question_id}:\nError: {forecast_summary.__class__.__name__} {forecast_summary}\nURL: https://www.metaculus.com/questions/{post_id}/\n"
            )
            errors.append(forecast_summary)
        else:
            print(forecast_summary)

    if errors:
        print("-----------------------------------------------\nErrors:\n")
        error_message = f"Errors were encountered: {errors}"
        print(error_message)
        raise RuntimeError(error_message)


######################## FINAL RUN #########################
if __name__ == "__main__":
    if USE_EXAMPLE_QUESTIONS:
        open_question_id_post_id = EXAMPLE_QUESTIONS
    else:
        open_question_id_post_id = get_open_question_ids_from_tournament()

    asyncio.run(
        forecast_questions(
            open_question_id_post_id,
            SUBMIT_PREDICTION,
            NUM_RUNS_PER_QUESTION,
            SKIP_PREVIOUSLY_FORECASTED_QUESTIONS,
        )
    )
