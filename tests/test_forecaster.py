from unittest.mock import AsyncMock, patch

import pytest
from forecasting_tools import MetaculusQuestion

from main import TemplateForecaster


@pytest.mark.asyncio
async def test_run_research_priority():
    """Test that research provider priority logic works correctly."""
    forecaster = TemplateForecaster(
        llms={
            "default": "mock_default_model",
            "parser": "mock_parser",
            "researcher": "mock_researcher",
            "summarizer": "mock_summarizer",
        }
    )
    question = MetaculusQuestion(question_text="Test question", page_url="http://example.com")

    # Test AskNews priority (highest priority when available)
    with patch("os.getenv") as mock_getenv:
        mock_getenv.side_effect = lambda x, default=None: {
            "ASKNEWS_CLIENT_ID": "asknews_client_id",
            "ASKNEWS_SECRET": "asknews_secret",
            "EXA_API_KEY": "exa_api_key",
            "PERPLEXITY_API_KEY": "perplexity_api_key",
            "OPENROUTER_API_KEY": "openrouter_api_key",
        }.get(x, default)

        with patch(
            "forecasting_tools.AskNewsSearcher.get_formatted_news_async", new_callable=AsyncMock
        ) as mock_asknews:
            mock_asknews.return_value = "AskNews Research"
            research = await forecaster.run_research(question)
            mock_asknews.assert_called_once_with(question.question_text)
            assert research == "AskNews Research"

    # Test Exa priority (when AskNews not available)
    with patch("os.getenv") as mock_getenv:
        mock_getenv.side_effect = lambda x, default=None: {
            "EXA_API_KEY": "exa_api_key",
            "PERPLEXITY_API_KEY": "perplexity_api_key",
            "OPENROUTER_API_KEY": "openrouter_api_key",
        }.get(x, default)

        with patch("main.TemplateForecaster._call_exa_smart_searcher", new_callable=AsyncMock) as mock_exa:
            mock_exa.return_value = "Exa Research"
            research = await forecaster.run_research(question)
            mock_exa.assert_called_once_with(question.question_text)
            assert research == "Exa Research"

    # Test Perplexity priority (when AskNews and Exa not available)
    with patch("os.getenv") as mock_getenv:
        mock_getenv.side_effect = lambda x, default=None: {
            "PERPLEXITY_API_KEY": "perplexity_api_key",
            "OPENROUTER_API_KEY": "openrouter_api_key",
        }.get(x, default)

        with patch("main.TemplateForecaster._call_perplexity", new_callable=AsyncMock) as mock_perplexity:
            mock_perplexity.return_value = "Perplexity Research"
            research = await forecaster.run_research(question)
            mock_perplexity.assert_called_once_with(question.question_text, use_open_router=False)
            assert research == "Perplexity Research"

    # Test OpenRouter priority (when only OpenRouter available)
    with patch("os.getenv") as mock_getenv:
        mock_getenv.side_effect = lambda x: {
            "OPENROUTER_API_KEY": "openrouter_api_key",
        }.get(x)

        with patch("main.TemplateForecaster._call_perplexity", new_callable=AsyncMock) as mock_openrouter:
            mock_openrouter.return_value = "OpenRouter Research"
            research = await forecaster.run_research(question)
            mock_openrouter.assert_called_once_with(question.question_text, use_open_router=True)
            assert research == "OpenRouter Research"

    # Test no research provider available
    with patch("os.getenv") as mock_getenv:
        mock_getenv.side_effect = lambda x: None

        research = await forecaster.run_research(question)
        assert research == ""
