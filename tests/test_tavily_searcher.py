import os
import unittest
from unittest.mock import patch

from bot.search_telemetry import (
    reset_search_provider_telemetry,
    snapshot_search_provider_telemetry,
)
from bot.tavily_searcher import (
    TavilySearcher,
    TavilySearchResult,
    TavilySmartSearcher,
)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    async def json(self) -> dict:
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _FailingResponse(_FakeResponse):
    def raise_for_status(self) -> None:
        raise RuntimeError("tavily failed")


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.post_calls: list[tuple[str, dict, dict]] = []

    def post(
        self, url: str, *, json: dict, headers: dict
    ):  # noqa: A002 - matches aiohttp signature
        self.post_calls.append((url, json, headers))
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class TestTavilySearcher(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        reset_search_provider_telemetry()

    async def asyncTearDown(self) -> None:
        reset_search_provider_telemetry()

    async def test_requires_api_key(self) -> None:
        previous = os.environ.pop("TAVILY_API_KEY", None)
        self.addCleanup(
            lambda: os.environ.__setitem__("TAVILY_API_KEY", previous)
            if previous is not None
            else os.environ.pop("TAVILY_API_KEY", None)
        )

        with self.assertRaises(ValueError):
            TavilySearcher()

    async def test_search_parses_results_and_sends_auth(self) -> None:
        previous = os.environ.get("TAVILY_API_KEY")
        os.environ["TAVILY_API_KEY"] = "tvly-test"
        self.addCleanup(
            lambda: os.environ.__setitem__("TAVILY_API_KEY", previous)
            if previous is not None
            else os.environ.pop("TAVILY_API_KEY", None)
        )

        fake_payload = {
            "results": [
                {
                    "title": "Result A",
                    "url": "https://example.com/a",
                    "content": "Snippet A",
                    "raw_content": None,
                    "score": 0.9,
                },
                {
                    "title": "Result B",
                    "url": "https://example.com/b",
                    "content": "Snippet B",
                    "raw_content": "Raw B",
                    "score": 0.7,
                },
            ]
        }
        response = _FakeResponse(fake_payload)
        session = _FakeSession(response)

        with patch("bot.tavily_searcher.aiohttp.ClientSession", return_value=session):
            searcher = TavilySearcher(
                timeout_seconds=5,
                search_depth="basic",
                topic="general",
                time_range="week",
                include_raw_content=True,
            )
            results = await searcher.search(query="hello", max_results=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].title, "Result A")
        self.assertEqual(results[0].url, "https://example.com/a")
        self.assertEqual(results[0].content, "Snippet A")

        self.assertEqual(len(session.post_calls), 1)
        url, json_body, headers = session.post_calls[0]
        self.assertEqual(url, "https://api.tavily.com/search")
        self.assertEqual(headers.get("Authorization"), "Bearer tvly-test")
        self.assertEqual(json_body.get("query"), "hello")
        self.assertEqual(json_body.get("max_results"), 2)
        self.assertEqual(json_body.get("include_raw_content"), True)
        self.assertEqual(json_body.get("time_range"), "week")

        telemetry = snapshot_search_provider_telemetry()
        self.assertEqual(telemetry["tavily"]["requests"], 1)
        self.assertEqual(telemetry["tavily"]["successes"], 1)
        self.assertEqual(telemetry["tavily"]["failures"], 0)
        self.assertEqual(telemetry["tavily"]["estimated_credits"], 1)
        self.assertEqual(telemetry["tavily"]["raw_content_requests"], 1)
        self.assertEqual(telemetry["tavily"]["basic_requests"], 1)

    async def test_search_records_failed_request(self) -> None:
        previous = os.environ.get("TAVILY_API_KEY")
        os.environ["TAVILY_API_KEY"] = "tvly-test"
        self.addCleanup(
            lambda: os.environ.__setitem__("TAVILY_API_KEY", previous)
            if previous is not None
            else os.environ.pop("TAVILY_API_KEY", None)
        )

        session = _FakeSession(_FailingResponse({}))

        with patch("bot.tavily_searcher.aiohttp.ClientSession", return_value=session):
            searcher = TavilySearcher(search_depth="advanced")
            with self.assertRaises(RuntimeError):
                await searcher.search(query="hello", max_results=3)

        telemetry = snapshot_search_provider_telemetry()
        self.assertEqual(telemetry["tavily"]["requests"], 1)
        self.assertEqual(telemetry["tavily"]["successes"], 0)
        self.assertEqual(telemetry["tavily"]["failures"], 1)
        self.assertEqual(telemetry["tavily"]["estimated_credits"], 2)
        self.assertEqual(telemetry["tavily"]["advanced_requests"], 1)


class _FakeWebPageParser:
    instances = []

    def __init__(self, *, limits, user_agent=None) -> None:
        self.limits = limits
        self.user_agent = user_agent
        self.urls = []
        self.closed = False
        _FakeWebPageParser.instances.append(self)

    async def get_clean_text(self, url: str) -> str:
        self.urls.append(url)
        return f"Extracted article body for {url}. " * 5

    async def close(self) -> None:
        self.closed = True


def _smart_searcher_for_extract(
    *,
    enabled: bool = True,
    min_chars: int = 50,
    max_urls: int = 1,
) -> TavilySmartSearcher:
    searcher = object.__new__(TavilySmartSearcher)
    searcher._extract_missing_content_enabled = enabled
    searcher._extract_min_content_chars = min_chars
    searcher._extract_max_urls = max_urls
    searcher._extract_timeout_seconds = 1
    searcher._per_result_char_budget = 200
    return searcher


class TestTavilySearchToExtract(unittest.IsolatedAsyncioTestCase):
    async def test_augments_only_limited_thin_results(self) -> None:
        _FakeWebPageParser.instances.clear()
        results = [
            TavilySearchResult(
                query="q",
                title="thin",
                url="https://example.com/thin",
                content="short",
                raw_content=None,
                score=0.9,
            ),
            TavilySearchResult(
                query="q",
                title="enough",
                url="https://example.com/enough",
                content="x" * 100,
                raw_content=None,
                score=0.8,
            ),
            TavilySearchResult(
                query="q",
                title="second thin",
                url="https://example.com/second-thin",
                content="small",
                raw_content=None,
                score=0.7,
            ),
        ]

        with patch("local_web_crawl.PlaywrightWebPageParser", _FakeWebPageParser):
            updated = await _smart_searcher_for_extract(
                max_urls=1
            )._extract_missing_content(results)

        self.assertEqual(len(_FakeWebPageParser.instances), 1)
        parser = _FakeWebPageParser.instances[0]
        self.assertEqual(parser.urls, ["https://example.com/thin"])
        self.assertTrue(parser.closed)
        self.assertIn("Extracted article body", updated[0].raw_content or "")
        self.assertIsNone(updated[1].raw_content)
        self.assertIsNone(updated[2].raw_content)

    async def test_returns_original_results_when_disabled(self) -> None:
        results = [
            TavilySearchResult(
                query="q",
                title="thin",
                url="https://example.com/thin",
                content="short",
                raw_content=None,
                score=0.9,
            )
        ]

        with patch("local_web_crawl.PlaywrightWebPageParser", _FakeWebPageParser):
            updated = await _smart_searcher_for_extract(
                enabled=False
            )._extract_missing_content(results)

        self.assertIs(updated, results)
