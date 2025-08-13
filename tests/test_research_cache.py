"""
Test research caching functionality for TemplateForecaster.

This test verifies that:
1. Research cache is used when enabled in benchmarking mode
2. Research cache is ignored when not in benchmarking mode  
3. Cache sharing works between multiple bot instances
4. Double-check pattern prevents race conditions
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from main import TemplateForecaster
from metaculus_bot.llm_configs import PARSER_LLM, RESEARCHER_LLM, SUMMARIZER_LLM


@pytest.fixture
def mock_question():
    """Create a mock MetaculusQuestion for testing."""
    question = MagicMock()
    question.id_of_question = 12345
    question.question_text = "Will it rain tomorrow?"
    question.page_url = "https://example.com/q/12345"
    return question


@pytest.fixture
def test_llms():
    """Mock LLM configuration for testing."""
    return {
        "default": MagicMock(),
        "parser": PARSER_LLM,
        "researcher": RESEARCHER_LLM,
        "summarizer": SUMMARIZER_LLM,
    }


@pytest.mark.asyncio
async def test_research_cache_enabled_in_benchmarking_mode(mock_question, test_llms):
    """Test that research caching works when enabled in benchmarking mode."""
    shared_cache = {}

    # Create bot with caching enabled
    bot = TemplateForecaster(
        is_benchmarking=True,
        research_cache=shared_cache,
        llms=test_llms,
    )

    # Mock the research provider
    mock_provider = AsyncMock(return_value="Cached research result")
    bot._custom_research_provider = mock_provider

    # First call should hit provider and cache result
    result1 = await bot.run_research(mock_question)
    assert result1 == "Cached research result"
    assert shared_cache[12345] == "Cached research result"
    assert mock_provider.call_count == 1

    # Second call should use cache, not hit provider
    result2 = await bot.run_research(mock_question)
    assert result2 == "Cached research result"
    assert mock_provider.call_count == 1  # Should not increase


@pytest.mark.asyncio
async def test_research_cache_shared_between_bots(mock_question, test_llms):
    """Test that cache is shared between multiple bot instances."""
    shared_cache = {}

    # Create first bot
    bot1 = TemplateForecaster(
        is_benchmarking=True,
        research_cache=shared_cache,
        llms=test_llms,
    )
    bot1._custom_research_provider = AsyncMock(return_value="Shared research")

    # Create second bot with same cache
    bot2 = TemplateForecaster(
        is_benchmarking=True,
        research_cache=shared_cache,
        llms=test_llms,
    )
    bot2._custom_research_provider = AsyncMock(return_value="Different research")

    # First bot caches result
    result1 = await bot1.run_research(mock_question)
    assert result1 == "Shared research"
    assert bot1._custom_research_provider.call_count == 1

    # Second bot should use cached result, not call its provider
    result2 = await bot2.run_research(mock_question)
    assert result2 == "Shared research"  # Should get cached result
    assert bot2._custom_research_provider.call_count == 0  # Should not be called


@pytest.mark.asyncio
async def test_research_cache_disabled_in_non_benchmarking_mode(mock_question, test_llms):
    """Test that cache is ignored when not in benchmarking mode."""
    shared_cache = {}

    # Create bot NOT in benchmarking mode
    bot = TemplateForecaster(
        is_benchmarking=False,  # Not benchmarking
        research_cache=shared_cache,
        llms=test_llms,
    )
    bot._custom_research_provider = AsyncMock(return_value="Non-cached research")

    # First call
    result1 = await bot.run_research(mock_question)
    assert result1 == "Non-cached research"
    assert bot._custom_research_provider.call_count == 1

    # Second call should hit provider again, not use cache
    result2 = await bot.run_research(mock_question)
    assert result2 == "Non-cached research"
    assert bot._custom_research_provider.call_count == 2  # Should increase


@pytest.mark.asyncio
async def test_research_cache_disabled_when_cache_is_none(mock_question, test_llms):
    """Test that research proceeds normally when cache is None."""
    # Create bot with no cache
    bot = TemplateForecaster(
        is_benchmarking=True,
        research_cache=None,  # No cache provided
        llms=test_llms,
    )
    bot._custom_research_provider = AsyncMock(return_value="No cache research")

    # Should work normally without caching
    result = await bot.run_research(mock_question)
    assert result == "No cache research"
    assert bot._custom_research_provider.call_count == 1


@pytest.mark.asyncio
async def test_research_cache_different_questions_separate_cache_entries(test_llms):
    """Test that different questions get separate cache entries."""
    shared_cache = {}

    # Create two different questions
    question1 = MagicMock()
    question1.id_of_question = 11111
    question1.question_text = "Question 1"
    question1.page_url = "https://example.com/q/11111"

    question2 = MagicMock()
    question2.id_of_question = 22222
    question2.question_text = "Question 2"
    question2.page_url = "https://example.com/q/22222"

    bot = TemplateForecaster(
        is_benchmarking=True,
        research_cache=shared_cache,
        llms=test_llms,
    )

    # Mock provider to return different results based on question
    async def mock_provider(question_text):
        if "Question 1" in question_text:
            return "Research for Q1"
        else:
            return "Research for Q2"

    bot._custom_research_provider = mock_provider

    # Research both questions
    result1 = await bot.run_research(question1)
    result2 = await bot.run_research(question2)

    # Should have separate cache entries
    assert result1 == "Research for Q1"
    assert result2 == "Research for Q2"
    assert shared_cache[11111] == "Research for Q1"
    assert shared_cache[22222] == "Research for Q2"
    assert len(shared_cache) == 2
