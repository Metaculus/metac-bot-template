"""Test that AskNews integration properly handles rate limiting."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from metaculus_bot.research_providers import _asknews_provider


@pytest.mark.asyncio
async def test_asknews_rate_limiting_delay():
    """Test that AskNews provider waits between API calls to respect rate limits."""

    # Mock the AsyncAskNewsSDK to track timing
    call_times = []

    async def mock_search_news(*args, **kwargs):
        call_times.append(asyncio.get_event_loop().time())
        # Create a minimal mock response
        mock_response = AsyncMock()
        mock_response.as_dicts = []
        return mock_response

    with patch("os.getenv") as mock_getenv:
        mock_getenv.side_effect = lambda key: {
            "ASKNEWS_CLIENT_ID": "test_client_id",
            "ASKNEWS_SECRET": "test_secret",
        }.get(key)

        with patch("asknews_sdk.AsyncAskNewsSDK") as mock_sdk_class:
            mock_sdk = AsyncMock()
            mock_sdk.news.search_news = mock_search_news
            mock_sdk_class.return_value.__aenter__.return_value = mock_sdk

            provider = _asknews_provider()
            await provider("test question")

            # Verify two calls were made
            assert len(call_times) == 2

            # Verify there was a delay between calls (should be ~1.2 seconds)
            time_diff = call_times[1] - call_times[0]
            assert time_diff >= 1.2, f"Expected delay >= 1.2s, got {time_diff:.2f}s"
            assert time_diff <= 6.0, f"Expected delay <= 6.0s, got {time_diff:.2f}s"


@pytest.mark.asyncio
async def test_asknews_calls_both_endpoints():
    """Test that AskNews provider calls both latest and historical news endpoints."""

    search_calls = []

    async def mock_search_news(*args, **kwargs):
        search_calls.append(kwargs.get("strategy", "unknown"))
        mock_response = AsyncMock()
        mock_response.as_dicts = []
        return mock_response

    with patch("os.getenv") as mock_getenv:
        mock_getenv.side_effect = lambda key: {
            "ASKNEWS_CLIENT_ID": "test_client_id",
            "ASKNEWS_SECRET": "test_secret",
        }.get(key)

        with patch("asknews_sdk.AsyncAskNewsSDK") as mock_sdk_class:
            mock_sdk = AsyncMock()
            mock_sdk.news.search_news = mock_search_news
            mock_sdk_class.return_value.__aenter__.return_value = mock_sdk

            provider = _asknews_provider()
            result = await provider("test question")

            # Verify both strategies were called
            assert "latest news" in search_calls
            assert "news knowledge" in search_calls
            assert len(search_calls) == 2

            # Verify result format
            assert "No articles were found" in result
