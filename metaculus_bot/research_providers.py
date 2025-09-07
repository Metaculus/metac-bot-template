"""Research provider strategy abstraction.

`choose_provider` returns an async callable that, given a question text, returns
formatted research.  The selection is governed by environment variables so the
logic lives in one place instead of being in `TemplateForecaster.run_research`.
"""

import asyncio
import logging
import os
import time
from typing import Any, Awaitable, Callable, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from forecasting_tools import GeneralLlm, SmartSearcher

from metaculus_bot.constants import (
    ASKNEWS_BACKOFF_SECS,
    ASKNEWS_MAX_CONCURRENCY,
    ASKNEWS_MAX_RPS,
    ASKNEWS_MAX_TRIES,
    RESEARCH_PROVIDER_ENV,
)

QuestionText = str
ResearchCallable = Callable[[QuestionText], Awaitable[str]]
logger = logging.getLogger(__name__)


class ResearchProvider(Protocol):
    async def __call__(self, question_text: str) -> str:  # pragma: no cover
        ...


# ---------------------------------------------------------------------------
# Concrete provider helpers
# ---------------------------------------------------------------------------


_ASKNEWS_GLOBAL_SEMAPHORE: asyncio.Semaphore | None = None
_ASKNEWS_RATE_LOCK: asyncio.Lock = asyncio.Lock()
_ASKNEWS_LAST_CALL_TS: float = 0.0


async def _asknews_rate_gate() -> None:
    global _ASKNEWS_LAST_CALL_TS
    if ASKNEWS_MAX_RPS <= 0:
        return
    min_interval = 1.0 / ASKNEWS_MAX_RPS
    async with _ASKNEWS_RATE_LOCK:
        now = time.monotonic()
        wait = _ASKNEWS_LAST_CALL_TS + min_interval - now
        if wait > 0:
            await asyncio.sleep(wait)
            now = time.monotonic()
        _ASKNEWS_LAST_CALL_TS = now


def _asknews_provider() -> ResearchCallable:
    global _ASKNEWS_GLOBAL_SEMAPHORE
    if _ASKNEWS_GLOBAL_SEMAPHORE is None:
        # Initialize a single global semaphore to throttle concurrency across all bots
        max_c = max(1, int(ASKNEWS_MAX_CONCURRENCY))
        _ASKNEWS_GLOBAL_SEMAPHORE = asyncio.Semaphore(max_c)

    async def _fetch(question_text: str) -> str:  # noqa: D401
        assert _ASKNEWS_GLOBAL_SEMAPHORE is not None
        tries = max(1, int(ASKNEWS_MAX_TRIES))
        backoff = float(ASKNEWS_BACKOFF_SECS)

        # Retry helper: only retry on known transient rate/concurrency errors
        def _is_retryable(err: Exception) -> bool:
            msg = str(err).lower()
            return ("429" in msg) or ("rate limit" in msg) or ("concurrency limit" in msg)

        async with _ASKNEWS_GLOBAL_SEMAPHORE:
            # Use custom AskNews integration with proper rate limiting between API calls
            from asknews_sdk import AsyncAskNewsSDK

            client_id = os.getenv("ASKNEWS_CLIENT_ID")
            secret = os.getenv("ASKNEWS_SECRET")
            if not client_id or not secret:
                raise ValueError("ASKNEWS_CLIENT_ID and ASKNEWS_SECRET environment variables must be set")

            logger.info(f"AskNews: Using custom integration, client_id={client_id[:8]}...")

            async with AsyncAskNewsSDK(
                client_id=client_id,
                client_secret=secret,
                scopes=set(["news"]),
            ) as sdk:
                # Phase 1: HOT (latest news) with its own retry loop, consuming from total budget
                hot_articles = None
                last_exc: Exception | None = None
                hot_attempt_used = 0
                # Hack: despite including rate limits in our asknews logic, we still get rate limits; manually massage addl waits to handle
                WAIT_FOR_HOT_SEC = 10.1
                logger.info(f"AskNews: Waiting {WAIT_FOR_HOT_SEC}s before hot news call...")
                await asyncio.sleep(WAIT_FOR_HOT_SEC)
                for hot_attempt in range(1, tries + 1):
                    hot_attempt_used = hot_attempt
                    try:
                        logger.info(f"AskNews HOT attempt {hot_attempt}/{tries}: Calling latest news...")
                        await _asknews_rate_gate()
                        hot_response = await sdk.news.search_news(
                            query=question_text,
                            n_articles=6,
                            return_type="both",
                            strategy="latest news",
                        )
                        hot_articles = hot_response.as_dicts
                        break
                    except Exception as e:
                        last_exc = e
                        if not _is_retryable(e):
                            raise
                        if hot_attempt < tries:
                            sleep_for = backoff * (10 + 3 ** (hot_attempt))
                            await asyncio.sleep(sleep_for)
                        else:
                            assert last_exc is not None
                            raise last_exc

                assert hot_articles is not None

                # Phase 2: HISTORICAL (news knowledge), reuse HOT results; do not re-call HOT on retries
                WAIT_FOR_HISTORICAL_SEC = 10.1
                logger.info(f"AskNews: Waiting {WAIT_FOR_HISTORICAL_SEC}s before historical news call...")
                await asyncio.sleep(WAIT_FOR_HISTORICAL_SEC)

                remaining_hist_tries = max(1, tries - (hot_attempt_used - 1))
                historical_articles = None
                for hist_attempt in range(1, remaining_hist_tries + 1):
                    try:
                        logger.info(
                            f"AskNews HIST attempt {hist_attempt}/{remaining_hist_tries}: Passing rate gate before historical news call..."
                        )
                        await _asknews_rate_gate()
                        logger.info(
                            f"AskNews HIST attempt {hist_attempt}/{remaining_hist_tries}: Calling historical news..."
                        )
                        historical_response = await sdk.news.search_news(
                            query=question_text,
                            n_articles=10,
                            return_type="both",
                            strategy="news knowledge",
                        )
                        historical_articles = historical_response.as_dicts
                        break
                    except Exception as e:
                        last_exc = e
                        if not _is_retryable(e):
                            raise
                        if hist_attempt < remaining_hist_tries:
                            sleep_for = backoff * (10 + 3 ** (hist_attempt))
                            await asyncio.sleep(sleep_for)
                        else:
                            assert last_exc is not None
                            raise last_exc

                assert historical_articles is not None

                # Combine and format articles like forecasting-tools does
                logger.info(
                    f"AskNews: Got {len(hot_articles)} hot articles, {len(historical_articles)} historical articles"
                )
                formatted_articles = "Here are the relevant news articles:\n\n"

                all_articles = []
                if hot_articles:
                    all_articles.extend(hot_articles)
                if historical_articles:
                    all_articles.extend(historical_articles)

                if not all_articles:
                    return "No articles were found for this query.\n\n"

                # URL deduplication (best-effort, order-preserving)
                try:
                    before = len(all_articles)
                    all_articles = _dedup_articles_by_url(all_articles)
                    removed = before - len(all_articles)
                    if removed > 0:
                        logger.info(
                            f"AskNews URL dedup: {before} -> {len(all_articles)} (removed {removed} duplicates)"
                        )
                except Exception as dedup_exc:  # pragma: no cover - protective
                    logger.warning(f"AskNews URL dedup failed; proceeding without dedup: {dedup_exc}")

                # Sort by date and format
                sorted_articles = sorted(all_articles, key=lambda x: x.pub_date, reverse=True)

                for article in sorted_articles:
                    pub_date = article.pub_date.strftime("%B %d, %Y %I:%M %p")
                    formatted_articles += f"**{article.eng_title}**\n{article.summary}\nOriginal language: {article.language}\nPublish date: {pub_date}\nSource:[{article.source_id}]({article.article_url})\n\n"

                logger.info(
                    f"AskNews: Success, got {len(formatted_articles)} chars from {len(hot_articles)} hot + {len(historical_articles)} historical articles"
                )
                return formatted_articles

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


def _perplexity_provider(use_open_router: bool = False, is_benchmarking: bool = False) -> ResearchCallable:
    async def _fetch(question_text: str) -> str:  # noqa: D401
        model_name = "openrouter/perplexity/sonar-reasoning-pro" if use_open_router else "perplexity/sonar-pro"
        model = GeneralLlm(model=model_name, temperature=0.1)
        # Exclude prediction markets research when benchmarking to avoid data leakage
        prediction_markets_instruction = (
            "" if is_benchmarking else "In addition to news, consider all relevant prediction markets.\n"
        )
        prompt = (
            "You are an assistant to a superforecaster.\n"
            "Generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.\n"
            f"{prediction_markets_instruction}"
            "Do not produce forecasts yourself. Provide data for the superforecaster.\n\n"
            f"Question:\n{question_text}"
        )
        return await model.invoke(prompt)

    return _fetch


# ---------------------------------------------------------------------------
# Strategy selector
# ---------------------------------------------------------------------------


def choose_provider_with_name(
    default_llm: GeneralLlm | None = None,
    exa_callback: ResearchCallable | None = None,
    perplexity_callback: ResearchCallable | None = None,
    openrouter_callback: ResearchCallable | None = None,
    is_benchmarking: bool = False,
) -> tuple[ResearchCallable, str]:
    """Return a research coroutine and its provider name.

    Priority order replicates pre-refactor behaviour:
    1. AskNews (ASKNEWS_CLIENT_ID & ASKNEWS_SECRET)
    2. Exa.ai (EXA_API_KEY)
    3. Perplexity (PERPLEXITY_API_KEY)
    4. Perplexity via OpenRouter (OPENROUTER_API_KEY)
    5. Fallback stub that returns an empty string.
    """
    forced = os.getenv(RESEARCH_PROVIDER_ENV)
    if forced:
        forced_lc = forced.strip().lower()
        if forced_lc == "asknews":
            # Fail fast if creds missing to make misconfig obvious
            if not (os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET")):
                raise ValueError("RESEARCH_PROVIDER=asknews requires ASKNEWS_CLIENT_ID and ASKNEWS_SECRET to be set")
            return _asknews_provider(), "asknews"
        if forced_lc == "exa":
            if exa_callback is not None:
                return exa_callback, "exa"
            if default_llm is None:
                raise ValueError("RESEARCH_PROVIDER=exa requires default_llm or exa_callback to be provided")
            return _exa_provider(default_llm), "exa"
        if forced_lc == "perplexity":
            if perplexity_callback is not None:
                return perplexity_callback, "perplexity"
            return _perplexity_provider(False, is_benchmarking), "perplexity"
        if forced_lc == "openrouter":
            if openrouter_callback is not None:
                return openrouter_callback, "openrouter"
            return _perplexity_provider(True, is_benchmarking), "openrouter"
        # Any other value behaves as auto

    if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
        return _asknews_provider(), "asknews"

    if os.getenv("EXA_API_KEY"):
        if exa_callback is not None:
            return exa_callback, "exa"
        if default_llm is None:
            raise ValueError("default_llm must be provided for Exa research provider")
        return _exa_provider(default_llm), "exa"

    if os.getenv("PERPLEXITY_API_KEY"):
        if perplexity_callback is not None:
            return perplexity_callback, "perplexity"
        return _perplexity_provider(False, is_benchmarking), "perplexity"

    if os.getenv("OPENROUTER_API_KEY"):
        if openrouter_callback is not None:
            return openrouter_callback, "openrouter"
        return _perplexity_provider(True, is_benchmarking), "openrouter"

    async def _empty(_: str) -> str:  # noqa: D401
        return ""

    return _empty, "none"


def choose_provider(
    default_llm: GeneralLlm | None = None,
    exa_callback: ResearchCallable | None = None,
    perplexity_callback: ResearchCallable | None = None,
    openrouter_callback: ResearchCallable | None = None,
    is_benchmarking: bool = False,
) -> ResearchCallable:
    provider, _ = choose_provider_with_name(
        default_llm=default_llm,
        exa_callback=exa_callback,
        perplexity_callback=perplexity_callback,
        openrouter_callback=openrouter_callback,
        is_benchmarking=is_benchmarking,
    )
    return provider


# ---------------------------------------------------------------------------
# URL normalization and dedup helpers (simple, robust, testable)
# ---------------------------------------------------------------------------


def _normalize_url_for_dedup(url: str) -> str:
    """Return a canonicalized URL for deduplication.

    - Lowercase scheme and netloc
    - Drop fragment
    - Remove common tracking params (utm_*, gclid, fbclid, igshid, ref, mc_cid, mc_eid)
    - Sort remaining query params
    - Strip single trailing slash on path
    - Normalize mobile and AMP variants (m. subdomain, trailing /amp)
    """
    if not url:
        return url
    parts = urlsplit(url)
    scheme = (parts.scheme or "").lower()
    netloc = (parts.netloc or "").lower()

    # Normalize mobile subdomain 'm.' to base domain when present
    if netloc.startswith("m."):
        netloc = netloc[2:]

    # Normalize path: drop trailing '/amp' and single trailing slash
    path = parts.path or ""
    if path.endswith("/amp"):
        path = path[:-4]
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    # Normalize query: remove tracking keys; sort remaining
    drop_keys = {"gclid", "fbclid", "igshid", "ref", "mc_cid", "mc_eid"}
    q = []
    for k, v in parse_qsl(parts.query, keep_blank_values=True):
        if k.startswith("utm_") or k in drop_keys:
            continue
        # Normalize trivial trailing slashes in parameter values (e.g., b=2/ -> b=2)
        if isinstance(v, str) and v.endswith("/"):
            v = v.rstrip("/")
        q.append((k, v))
    # Sort for canonical order
    q.sort()
    query = urlencode(q, doseq=True)

    # Drop fragment
    fragment = ""

    return urlunsplit((scheme, netloc, path, query, fragment))


def _dedup_articles_by_url(articles: list[Any]) -> list[Any]:
    """Order-preserving deduplication of articles by normalized URL.

    Articles may be objects with attribute `article_url` or dicts with key `article_url`.
    Items without a URL are kept.
    """
    seen: set[str] = set()
    result: list[Any] = []
    for item in articles:
        url = None
        # Attribute access first, then mapping
        if hasattr(item, "article_url"):
            url = getattr(item, "article_url")
        elif isinstance(item, dict):  # type: ignore[unreachable]
            url = item.get("article_url")

        if not url:
            result.append(item)
            continue

        norm = _normalize_url_for_dedup(str(url))
        if norm in seen:
            continue
        seen.add(norm)
        result.append(item)

    # Defensive: ensure not all items were dropped; if so, keep the first
    if not result and articles:
        result.append(articles[0])
    return result
