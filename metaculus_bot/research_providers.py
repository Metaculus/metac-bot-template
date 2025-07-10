from __future__ import annotations

"""Research provider strategy abstraction.

`choose_provider` returns an async callable that, given a question text, returns
formatted research.  The selection is governed by environment variables so the
logic lives in one place instead of being in `TemplateForecaster.run_research`.
"""

import os
from typing import Awaitable, Callable, Protocol

from forecasting_tools import AskNewsSearcher, SmartSearcher, GeneralLlm

QuestionText = str
ResearchCallable = Callable[[QuestionText], Awaitable[str]]


class ResearchProvider(Protocol):
    async def __call__(self, question_text: str) -> str:  # pragma: no cover
        ...


# ---------------------------------------------------------------------------
# Concrete provider helpers
# ---------------------------------------------------------------------------


def _asknews_provider() -> ResearchCallable:
    async def _fetch(question_text: str) -> str:  # noqa: D401
        return await AskNewsSearcher().get_formatted_news_async(question_text)

    return _fetch


def _exa_provider(default_llm: GeneralLlm) -> ResearchCallable:
    async def _fetch(question_text: str) -> str:  # noqa: D401
        searcher = SmartSearcher(
            model=default_llm,
            temperature=0,
            num_searches_to_run=2,
            num_sites_per_search=10,
        )
        prompt = (
            "You are an assistant to a superforecaster. The superforecaster will give"
            " you a question they intend to forecast on. To be a great assistant, you generate"
            " a concise but detailed rundown of the most relevant news, including if the question"
            " would resolve Yes or No based on current information. You do not produce forecasts yourself."
            f"\n\nThe question is: {question_text}"
        )
        return await searcher.invoke(prompt)

    return _fetch


def _perplexity_provider(use_open_router: bool = False) -> ResearchCallable:
    async def _fetch(question_text: str) -> str:  # noqa: D401
        model_name = (
            "openrouter/perplexity/sonar-reasoning-pro" if use_open_router else "perplexity/sonar-pro"
        )
        model = GeneralLlm(model=model_name, temperature=0.1)
        prompt = (
            "You are an assistant to a superforecaster.\n"
            "Generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.\n"
            "In addition to news, consider all relevant prediction markets.\n"
            "Do not produce forecasts yourself. Provide data for the superforecaster.\n\n"
            f"Question:\n{question_text}"
        )
        return await model.invoke(prompt)

    return _fetch


# ---------------------------------------------------------------------------
# Strategy selector
# ---------------------------------------------------------------------------

def choose_provider(
    default_llm: GeneralLlm | None = None,
    exa_callback: ResearchCallable | None = None,
    perplexity_callback: ResearchCallable | None = None,
    openrouter_callback: ResearchCallable | None = None,
) -> ResearchCallable:
    """Return a research-fetching coroutine based on env vars.

    Priority order replicates pre-refactor behaviour:
    1. AskNews (ASKNEWS_CLIENT_ID & ASKNEWS_SECRET)
    2. Exa.ai (EXA_API_KEY)
    3. Perplexity (PERPLEXITY_API_KEY)
    4. Perplexity via OpenRouter (OPENROUTER_API_KEY)
    5. Fallback stub that returns an empty string.
    """

    if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
        return _asknews_provider()

    if os.getenv("EXA_API_KEY"):
        if exa_callback is not None:
            return exa_callback
        if default_llm is None:
            raise ValueError("default_llm must be provided for Exa research provider")
        return _exa_provider(default_llm)

    if os.getenv("PERPLEXITY_API_KEY"):
        if perplexity_callback is not None:
            return perplexity_callback
        return _perplexity_provider(False)

    if os.getenv("OPENROUTER_API_KEY"):
        if openrouter_callback is not None:
            return openrouter_callback
        return _perplexity_provider(True)

    async def _empty(_: str) -> str:  # noqa: D401
        return ""

    return _empty 