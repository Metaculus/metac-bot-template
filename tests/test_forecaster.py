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

        # Mock the provider function returned by choose_provider_with_name
        mock_asknews_func = AsyncMock(return_value="AskNews Research")
        with patch("main.choose_provider_with_name") as mock_choose:
            mock_choose.return_value = (mock_asknews_func, "asknews")

            research = await forecaster.run_research(question)
            mock_asknews_func.assert_called_once_with(question.question_text)
            assert research == "AskNews Research"

    # Test Exa priority (when AskNews not available)
    with patch("os.getenv") as mock_getenv:
        mock_getenv.side_effect = lambda x, default=None: {
            "EXA_API_KEY": "exa_api_key",
            "PERPLEXITY_API_KEY": "perplexity_api_key",
            "OPENROUTER_API_KEY": "openrouter_api_key",
        }.get(x, default)

        # Mock the provider function returned by choose_provider_with_name
        mock_exa_func = AsyncMock(return_value="Exa Research")
        with patch("main.choose_provider_with_name") as mock_choose:
            mock_choose.return_value = (mock_exa_func, "exa")

            research = await forecaster.run_research(question)
            mock_exa_func.assert_called_once_with(question.question_text)
            assert research == "Exa Research"

    # Test Perplexity priority (when AskNews and Exa not available)
    with patch("os.getenv") as mock_getenv:
        mock_getenv.side_effect = lambda x, default=None: {
            "PERPLEXITY_API_KEY": "perplexity_api_key",
            "OPENROUTER_API_KEY": "openrouter_api_key",
        }.get(x, default)

        # Mock the provider function returned by choose_provider_with_name
        mock_perplexity_func = AsyncMock(return_value="Perplexity Research")
        with patch("main.choose_provider_with_name") as mock_choose:
            mock_choose.return_value = (mock_perplexity_func, "perplexity")

            research = await forecaster.run_research(question)
            mock_perplexity_func.assert_called_once_with(question.question_text)
            assert research == "Perplexity Research"

    # Test OpenRouter priority (when only OpenRouter available)
    with patch("os.getenv") as mock_getenv:
        mock_getenv.side_effect = lambda x, default=None: {
            "OPENROUTER_API_KEY": "openrouter_api_key",
        }.get(x, default)

        # Mock the provider function returned by choose_provider_with_name
        mock_openrouter_func = AsyncMock(return_value="OpenRouter Research")
        with patch("main.choose_provider_with_name") as mock_choose:
            mock_choose.return_value = (mock_openrouter_func, "openrouter")

            research = await forecaster.run_research(question)
            mock_openrouter_func.assert_called_once_with(question.question_text)
            assert research == "OpenRouter Research"

    # Test no research provider available
    with patch("os.getenv") as mock_getenv:
        mock_getenv.side_effect = lambda x, default=None: default

        # Mock the provider function to return empty string fallback
        mock_empty_func = AsyncMock(return_value="")
        with patch("main.choose_provider_with_name") as mock_choose:
            mock_choose.return_value = (mock_empty_func, "fallback")

            research = await forecaster.run_research(question)
            mock_empty_func.assert_called_once_with(question.question_text)
            assert research == ""
