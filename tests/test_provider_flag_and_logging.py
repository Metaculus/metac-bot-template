from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import pytest
from forecasting_tools import MetaculusQuestion

from main import TemplateForecaster


@pytest.mark.asyncio
async def test_research_provider_flag_and_logging(mock_os_getenv, caplog):
    # Force AskNews via env flag and provide required creds
    mock_os_getenv.side_effect = lambda x: {
        "RESEARCH_PROVIDER": "asknews",
        "ASKNEWS_CLIENT_ID": "client",
        "ASKNEWS_SECRET": "secret",
    }.get(x)

    bot = TemplateForecaster(
        llms={
            "default": "mock_default_model",
            "parser": "mock_parser",
            "researcher": "mock_researcher",
            "summarizer": "mock_summarizer",
        }
    )
    q = MetaculusQuestion(question_text="Test", page_url="http://example.com")

    with patch(
        "forecasting_tools.AskNewsSearcher.get_formatted_news_async",
        new_callable=AsyncMock,
    ) as mock_ask:
        mock_ask.return_value = "AskNews Research"
        with caplog.at_level(logging.INFO):
            res = await bot.run_research(q)
        assert res == "AskNews Research"
        assert any("Using research provider: asknews" in rec.message for rec in caplog.records)
