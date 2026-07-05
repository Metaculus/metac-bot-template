from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, replace
from datetime import datetime

import aiohttp

from forecasting_tools import GeneralLlm, clean_indents
from forecasting_tools.util.misc import fill_in_citations

from bot.search_telemetry import record_tavily_search_request
from bot.source_quality import format_source_quality_table

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TavilySearchResult:
    query: str
    title: str | None
    url: str | None
    content: str | None
    raw_content: str | None
    score: float | None


class TavilySearcher:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 30,
        max_concurrency: int = 3,
        search_depth: str = "basic",
        topic: str = "general",
        time_range: str | None = None,
        include_raw_content: bool = False,
        include_images: bool = False,
    ) -> None:
        self._api_key = (api_key or os.getenv("TAVILY_API_KEY") or "").strip()
        if not self._api_key:
            raise ValueError("TAVILY_API_KEY is not set in the environment variables")

        self._timeout = aiohttp.ClientTimeout(total=max(1.0, float(timeout_seconds)))
        self._sem = asyncio.Semaphore(max(1, int(max_concurrency)))

        self._search_depth = (search_depth or "basic").strip()
        self._topic = (topic or "general").strip()
        self._time_range = (time_range or "").strip() or None
        self._include_raw_content = bool(include_raw_content)
        self._include_images = bool(include_images)

    async def search(
        self,
        *,
        query: str,
        max_results: int,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> list[TavilySearchResult]:
        query = (query or "").strip()
        if not query:
            return []

        url = "https://api.tavily.com/search"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload: dict = {
            "query": query,
            "search_depth": self._search_depth,
            "topic": self._topic,
            "max_results": max(1, int(max_results)),
            "include_answer": False,
            "include_raw_content": self._include_raw_content,
            "include_images": self._include_images,
            "include_domains": include_domains or [],
            "exclude_domains": exclude_domains or [],
        }
        if self._time_range:
            payload["time_range"] = self._time_range

        data: dict = {}
        success = False
        try:
            async with self._sem:
                async with aiohttp.ClientSession(timeout=self._timeout) as session:
                    async with session.post(
                        url, json=payload, headers=headers
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()
                        success = True
        finally:
            record_tavily_search_request(
                search_depth=self._search_depth,
                max_results=max(1, int(max_results)),
                include_raw_content=self._include_raw_content,
                success=success,
            )

        results: list[TavilySearchResult] = []
        for item in data.get("results") or []:
            if not isinstance(item, dict):
                continue
            results.append(
                TavilySearchResult(
                    query=query,
                    title=item.get("title"),
                    url=item.get("url"),
                    content=item.get("content"),
                    raw_content=item.get("raw_content"),
                    score=item.get("score"),
                )
            )
        return results


class TavilySmartSearcher:
    """
    SmartSearcher-like research workflow powered by Tavily.
    """

    def __init__(
        self,
        *,
        model: str | GeneralLlm,
        temperature: float | None = None,
        num_searches_to_run: int = 1,
        num_sites_per_search: int = 5,
        search_depth: str = "basic",
        topic: str = "general",
        time_range: str | None = None,
        include_raw_content: bool = False,
        per_result_char_budget: int = 1200,
        total_context_char_budget: int = 12000,
        timeout_seconds: float = 30,
        extract_missing_content: bool = False,
        extract_min_content_chars: int = 500,
        extract_max_urls: int = 2,
        extract_timeout_seconds: float = 20,
    ) -> None:
        if temperature is not None and not (0 <= temperature <= 1):
            raise ValueError("temperature must be between 0 and 1")
        self._num_searches_to_run = max(1, int(num_searches_to_run))
        self._num_sites_per_search = max(1, int(num_sites_per_search))
        self._per_result_char_budget = max(200, int(per_result_char_budget))
        self._total_context_char_budget = max(1000, int(total_context_char_budget))
        self._extract_missing_content_enabled = bool(extract_missing_content)
        self._extract_min_content_chars = max(0, int(extract_min_content_chars))
        self._extract_max_urls = max(0, int(extract_max_urls))
        self._extract_timeout_seconds = max(1, int(extract_timeout_seconds))

        self._llm = (
            model
            if isinstance(model, GeneralLlm)
            else GeneralLlm(model=model, temperature=temperature)
        )
        self._tavily = TavilySearcher(
            timeout_seconds=timeout_seconds,
            search_depth=search_depth,
            topic=topic,
            time_range=time_range,
            include_raw_content=include_raw_content,
        )

    async def invoke(self, prompt: str) -> str:
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        search_terms = await self._come_up_with_search_queries(prompt)
        results = await self._search(search_terms)
        results = await self._extract_missing_content(results)
        logger.info(
            "TavilySmartSearcher completed searches: searches=%s deduped_results=%s",
            len(search_terms),
            len(results),
        )
        report = await self._compile_report(results, prompt)
        return self._link_citations(report, results)

    async def _come_up_with_search_queries(self, prompt: str) -> list[str]:
        prompt = clean_indents(
            f"""
            You have been given the following instructions. Instructions are included between <><><><><><><><><><><><> tags.

            <><><><><><><><><><><><>
            {prompt}
            <><><><><><><><><><><><>

            Generate {self._num_searches_to_run} web searches that will help you fulfill any questions in the instructions.
            Make them each target different aspects of the question.
            Please provide the searches as a JSON list of strings like this:
            ["search 1", "search 2"]
            Give no other text than the list of search terms.
            """
        )
        search_terms = await self._llm.invoke_and_return_verified_type(prompt, list[str])
        cleaned = [(term or "").strip() for term in search_terms if (term or "").strip()]
        return cleaned[: self._num_searches_to_run]

    async def _search(self, search_terms: list[str]) -> list[TavilySearchResult]:
        if not search_terms:
            return []

        batches = await asyncio.gather(
            *[
                self._tavily.search(query=term, max_results=self._num_sites_per_search)
                for term in search_terms
            ]
        )
        flattened = [item for batch in batches for item in batch]

        seen: set[str] = set()
        deduped: list[TavilySearchResult] = []
        for item in flattened:
            url = (item.url or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            deduped.append(item)
        return deduped

    async def _extract_missing_content(
        self, results: list[TavilySearchResult]
    ) -> list[TavilySearchResult]:
        if (
            not self._extract_missing_content_enabled
            or self._extract_max_urls <= 0
            or self._extract_min_content_chars <= 0
            or not results
        ):
            return results

        candidates: list[tuple[int, TavilySearchResult]] = []
        for idx, item in enumerate(results):
            url = (item.url or "").strip()
            if not (url.startswith("http://") or url.startswith("https://")):
                continue
            body = (item.raw_content or item.content or "").strip()
            if len(body) >= self._extract_min_content_chars:
                continue
            candidates.append((idx, item))
            if len(candidates) >= self._extract_max_urls:
                break

        if not candidates:
            return results

        try:
            from local_web_crawl import LocalCrawlLimits, PlaywrightWebPageParser
        except Exception as e:
            logger.info("Tavily Search-to-Extract unavailable: %s", e)
            return results

        per_url_budget = max(
            self._per_result_char_budget, self._extract_min_content_chars
        )
        limits = LocalCrawlLimits(
            max_urls=len(candidates),
            max_concurrency=min(2, len(candidates)),
            navigation_timeout_seconds=self._extract_timeout_seconds,
            network_idle_timeout_seconds=max(
                1, min(5, self._extract_timeout_seconds)
            ),
            total_char_budget=per_url_budget * len(candidates),
            per_url_char_budget=per_url_budget,
        )
        user_agent = os.getenv("BOT_LOCAL_CRAWL_USER_AGENT", "").strip() or None
        parser = PlaywrightWebPageParser(limits=limits, user_agent=user_agent)

        async def fetch_one(
            idx: int, item: TavilySearchResult
        ) -> tuple[int, str | None]:
            url = (item.url or "").strip()
            try:
                return idx, await parser.get_clean_text(url)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.info("Tavily Search-to-Extract failed for %s: %s", url, e)
                return idx, None

        try:
            fetched = await asyncio.gather(
                *[fetch_one(idx, item) for idx, item in candidates]
            )
        finally:
            await parser.close()

        updated = list(results)
        extracted_count = 0
        for idx, extracted in fetched:
            extracted = (extracted or "").strip()
            if not extracted:
                continue
            existing = (updated[idx].raw_content or updated[idx].content or "").strip()
            if len(extracted) <= len(existing):
                continue
            updated[idx] = replace(updated[idx], raw_content=extracted)
            extracted_count += 1

        if extracted_count:
            logger.info(
                "Tavily Search-to-Extract augmented %s/%s thin results",
                extracted_count,
                len(candidates),
            )

        return updated

    async def _compile_report(
        self, results: list[TavilySearchResult], original_instructions: str
    ) -> str:
        if not results:
            return "No search results found for the query."

        context_lines: list[str] = []
        remaining = self._total_context_char_budget
        for idx, item in enumerate(results, start=1):
            url = item.url or "No URL found"
            title = item.title or "Untitled"
            body = (item.raw_content or item.content or "").strip()
            if not body:
                continue
            if remaining <= 0:
                break
            body = body[: min(len(body), self._per_result_char_budget, remaining)]
            remaining -= len(body)
            context_lines.append(
                f'[{idx}] "{body}". [Source: {url} titled "{title}"]'
            )

        search_result_context = "\n".join(context_lines).strip()
        if not search_result_context:
            return "No usable search results found for the query."

        source_quality_table = format_source_quality_table(results)

        prompt = clean_indents(
            f"""
            Today is {datetime.now().strftime("%Y-%m-%d")}.
            You have been given the following instructions. Instructions are included between <><><><><><><><><><><><> tags.

            <><><><><><><><><><><><>
            {original_instructions}
            <><><><><><><><><><><><>

            After searching the internet, you found the following results. Results are included between <><><><><><><><><><><><> tags.
            <><><><><><><><><><><><>
            {search_result_context}
            <><><><><><><><><><><><>

            Source quality assessment:
            <><><><><><><><><><><><>
            {source_quality_table}
            <><><><><><><><><><><><>

            Please follow the instructions and use the search results to answer the question. Unless the instructions specify otherwise, cite your sources inline using [1], [2], etc and use markdown formatting.
            Treat low-reliability or social/forum sources as leads rather than decisive evidence. Prefer official, primary, academic, and directly resolution-relevant sources when they conflict with secondary sources.
            """
        )
        return await self._llm.invoke(prompt)

    @staticmethod
    def _link_citations(text: str, results: list[TavilySearchResult]) -> str:
        urls: list[str] = []
        for item in results:
            url = (item.url or "").strip()
            if not url:
                url = "No URL found"
            urls.append(url)
        return fill_in_citations(urls, text, use_citation_brackets=False)
