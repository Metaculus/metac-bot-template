from unittest.mock import patch

import pytest
from forecasting_tools import MetaculusQuestion

from main import TemplateForecaster


@pytest.mark.asyncio
async def test_run_research_prefers_custom_provider(mock_os_getenv):
    async def custom_provider(_: str) -> str:  # noqa: D401
        return "Custom Research"

    bot = TemplateForecaster(
        llms={
            "default": "mock",
            "parser": "mock",
            "researcher": "mock",
            "summarizer": "mock",
        },
        research_provider=custom_provider,
    )
    q = MetaculusQuestion(question_text="Test", page_url="http://example.com")

    # Ensure environment would otherwise enable other providers
    mock_os_getenv.side_effect = lambda x: {
        "ASKNEWS_CLIENT_ID": "x",
        "ASKNEWS_SECRET": "y",
        "EXA_API_KEY": "z",
        "PERPLEXITY_API_KEY": "p",
        "OPENROUTER_API_KEY": "o",
    }.get(x)

    # Patch underlying providers to ensure they would be called if not overridden
    with patch("forecasting_tools.AskNewsSearcher.get_formatted_news_async") as asknews_mock:
        asknews_mock.return_value = "Should not be used"
        res = await bot.run_research(q)
        assert res == "Custom Research"
